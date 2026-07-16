using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using System.IO;
using System.Net.Sockets;
using System.Text;

namespace IdaHavuzTesti
{
    class Program
    {
        // train.ipynb'deki model.names sırasına göre:
        // black, green, orange, red, yellow
        static readonly string[] ClassNames = { "black", "green", "orange", "red", "yellow" };

        // Avalonia test arayüzüne TCP gönderim
        const string UiIp = "127.0.0.1";
        const int UiPort = 5055;

        const int InputSize = 640;
        const float ConfThreshold = 0.5f;
        const float NmsThreshold = 0.45f;

        // Mesafe tahmini için yaklaşık kamera görüş açıları
        // Daha sonra gerçek ölçümle kalibre edilebilir.
        const double CameraHFovDeg = 62.0;
        const double CameraVFovDeg = 49.0;

        static void Main(string[] args)
        {
            Console.WriteLine("Havuz Testi Baslatiliyor...");

            TcpClient? uiClient = null;
            StreamWriter? uiWriter = null;

            // 1. Modelin yüklenmesi
            string modelPath = "best.onnx";
            using var session = new InferenceSession(modelPath);

            // 2. Kamera bağlantısı
            using var capture = new VideoCapture("/dev/video4", VideoCaptureAPIs.V4L2);

            capture.Set(VideoCaptureProperties.FourCC,
                VideoWriter.FourCC('Y','U','Y','V'));

            capture.Set(VideoCaptureProperties.FrameWidth, 640);
            capture.Set(VideoCaptureProperties.FrameHeight, 480);
            capture.Set(VideoCaptureProperties.Fps, 30);

            Console.WriteLine($"Kamera açıldı: {capture.IsOpened()}");
            if (!capture.IsOpened())
            {
                Console.WriteLine("HATA: Kamera acilamadi. Baglantiyi kontrol edin.");
                return;
            }

            capture.Set(VideoCaptureProperties.FrameWidth, 640);
            capture.Set(VideoCaptureProperties.FrameHeight, 480);

            // 3. Video kaydedici
            var fourcc = VideoWriter.FourCC('m', 'p', '4', 'v');
            using var writer = new VideoWriter("havuz_testi_kayit.mp4", fourcc, 10, new Size(640, 480));

            using var frame = new Mat();

            try
            {
                while (true)
                {
                    capture.Read(frame);

                    if (frame.Empty())
                        break;

                    // 4. Ön işleme
                    using Mat processedFrame = ApplyGammaCorrection(frame, 1.2);

                    // 5. YOLOv8 ONNX çıkarımı
                    var detections = RunYoloInference(session, processedFrame);

                    // 6. Arayüze TCP ile gönder
                    EnsureUiConnection(ref uiClient, ref uiWriter);
                    SendVisionFrame(uiWriter, detections, processedFrame.Width, processedFrame.Height);

                    // 7. Terminal çıktısı
                    foreach (var det in detections)
                    {
                        Console.WriteLine(
                            $"DETECT,{det.ClassName},{det.Confidence:F2},X={det.X},Y={det.Y},W={det.Width},H={det.Height},DIST={det.DistanceM:F2}m,LATERAL={det.LateralM:F2}m"
                        );
                    }

                    Console.Out.Flush();

                    // 8. Tespitleri görüntüye çiz
                    DrawDetections(processedFrame, detections);

                    // 9. Göster ve kaydet
                    writer.Write(processedFrame);
                    Cv2.ImShow("IDA Havuz Testi Gorusu", processedFrame);

                    if (Cv2.WaitKey(1) == 'q')
                        break;
                }
            }
            finally
            {
                capture.Release();
                writer.Release();
                Cv2.DestroyAllWindows();
            }
        }

        // --- Gamma düzeltmesi ---
        static Mat ApplyGammaCorrection(Mat original, double gamma)
        {
            using Mat floatImage = new Mat();

            original.ConvertTo(floatImage, MatType.CV_32F, 1.0 / 255.0);
            Cv2.Pow(floatImage, gamma, floatImage);

            Mat result = new Mat();
            floatImage.ConvertTo(result, MatType.CV_8U, 255.0);

            return result;
        }

        // --- YOLOv8 ONNX çıkarımı ---
        static List<Detection> RunYoloInference(InferenceSession session, Mat frame)
        {
            int origW = frame.Width;
            int origH = frame.Height;

            // 5a. Letterbox: oran bozmadan 640x640'a yerleştir
            float scale = Math.Min((float)InputSize / origW, (float)InputSize / origH);

            int newW = (int)Math.Round(origW * scale);
            int newH = (int)Math.Round(origH * scale);

            int padX = (InputSize - newW) / 2;
            int padY = (InputSize - newH) / 2;

            using Mat resized = new Mat();
            Cv2.Resize(frame, resized, new Size(newW, newH));

            using Mat letterboxed = new Mat(InputSize, InputSize, MatType.CV_8UC3, Scalar.All(114));

            using (Mat roi = new Mat(letterboxed, new Rect(padX, padY, newW, newH)))
            {
                resized.CopyTo(roi);
            }

            // 5b. BGR -> RGB, HWC -> CHW, normalize [0,1]
            using Mat rgb = new Mat();
            Cv2.CvtColor(letterboxed, rgb, ColorConversionCodes.BGR2RGB);

            var inputTensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

            for (int y = 0; y < InputSize; y++)
            {
                for (int x = 0; x < InputSize; x++)
                {
                    Vec3b px = rgb.At<Vec3b>(y, x);

                    inputTensor[0, 0, y, x] = px.Item0 / 255f; // R
                    inputTensor[0, 1, y, x] = px.Item1 / 255f; // G
                    inputTensor[0, 2, y, x] = px.Item2 / 255f; // B
                }
            }

            string inputName = session.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            // 5c. Modeli çalıştır
            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();

            // Beklenen şekil: [1, 4 + numClasses, N]
            int channels = output.Dimensions[1];
            int numBoxes = output.Dimensions[2];
            int numClasses = channels - 4;

            var boxes = new List<Rect2d>();
            var scores = new List<float>();
            var classIds = new List<int>();

            for (int i = 0; i < numBoxes; i++)
            {
                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

                float bestScore = 0f;
                int bestClass = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    float score = output[0, 4 + c, i];

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                if (bestScore < ConfThreshold)
                    continue;

                float x1 = cx - w / 2f;
                float y1 = cy - h / 2f;

                boxes.Add(new Rect2d(x1, y1, w, h));
                scores.Add(bestScore);
                classIds.Add(bestClass);
            }

            // 5d. NMS
            CvDnn.NMSBoxes(boxes, scores, ConfThreshold, NmsThreshold, out int[] keepIndices);

            var detections = new List<Detection>();

            foreach (int idx in keepIndices)
            {
                var box = boxes[idx];

                // Letterbox paddingini çıkar ve orijinal frame boyutuna dön
                float x1 = (float)((box.X - padX) / scale);
                float y1 = (float)((box.Y - padY) / scale);
                float w = (float)(box.Width / scale);
                float h = (float)(box.Height / scale);

                x1 = Math.Clamp(x1, 0, origW - 1);
                y1 = Math.Clamp(y1, 0, origH - 1);
                w = Math.Clamp(w, 0, origW - x1);
                h = Math.Clamp(h, 0, origH - y1);

                int classId = classIds[idx];

                string className = classId >= 0 && classId < ClassNames.Length
                    ? ClassNames[classId]
                    : $"Sinif_{classId}";

                var detection = new Detection
                {
                    X = (int)x1,
                    Y = (int)y1,
                    Width = (int)w,
                    Height = (int)h,
                    ClassName = className,
                    Confidence = scores[idx]
                };

                detection.DistanceM = EstimateDistanceM(detection, origH);
                detection.LateralM = EstimateLateralM(detection, origW, detection.DistanceM);

                detections.Add(detection);
            }

            return detections;
        }

        // --- TCP bağlantısı ---
        static void EnsureUiConnection(ref TcpClient? client, ref StreamWriter? writer)
        {
            if (client != null && client.Connected && writer != null)
                return;

            try
            {
                client?.Close();

                client = new TcpClient();
                client.Connect(UiIp, UiPort);

                writer = new StreamWriter(client.GetStream(), Encoding.UTF8)
                {
                    AutoFlush = true
                };

                Console.WriteLine($"Arayüze bağlandı: {UiIp}:{UiPort}");
            }
            catch
            {
                writer = null;
                client = null;
            }
        }

        // --- Arayüze veri gönderimi ---
        static void SendVisionFrame(StreamWriter? writer, List<Detection> detections, int frameW, int frameH)
        {
            if (writer == null)
                return;

            StringBuilder sb = new StringBuilder();

            // Format:
            // FRAME,640,480;yellow,0.91,250,130,80,120,2.40,-0.30
            // FRAME,frameW,frameH;color,conf,x,y,w,h,distanceM,lateralM
            sb.Append($"FRAME,{frameW},{frameH}");

            foreach (var det in detections)
            {
                sb.Append(
                    FormattableString.Invariant(
                        $";{det.ClassName},{det.Confidence:F2},{det.X},{det.Y},{det.Width},{det.Height},{det.DistanceM:F2},{det.LateralM:F2}"
                    )
                );
            }

            try
            {
                writer.WriteLine(sb.ToString());
            }
            catch
            {
                // Arayüz kapanırsa görüntü işleme programı çökmesin.
            }
        }

        // --- Mesafe tahmini ---
        static double EstimateDistanceM(Detection det, int frameH)
        {
            if (det.Height <= 0)
                return 0;

            double realHeightM = GetRealHeightM(det.ClassName);

            double fovRad = DegToRad(CameraVFovDeg);
            double focalPx = frameH / (2.0 * Math.Tan(fovRad / 2.0));

            double distanceM = (realHeightM * focalPx) / det.Height;

            return Math.Clamp(distanceM, 0.1, 50.0);
        }

        // --- Sağ/sol konum tahmini ---
        static double EstimateLateralM(Detection det, int frameW, double distanceM)
        {
            double centerX = det.X + det.Width / 2.0;
            double normalizedX = (centerX - frameW / 2.0) / (frameW / 2.0);

            double halfFovRad = DegToRad(CameraHFovDeg) / 2.0;
            double angleRad = Math.Atan(normalizedX * Math.Tan(halfFovRad));

            return distanceM * Math.Tan(angleRad);
        }

        // --- Sınıfa göre gerçek duba yüksekliği ---
        static double GetRealHeightM(string className)
        {
            return className.ToLowerInvariant() switch
            {
                // Engel / parkur dubaları
                "orange" => 0.50,
                "yellow" => 0.50,

                // Hedef dubaları
                "black" => 0.95,
                "red" => 0.95,
                "green" => 0.95,

                _ => 0.50
            };
        }

        static double DegToRad(double deg)
        {
            return deg * Math.PI / 180.0;
        }

        // --- Tespitleri kamera görüntüsüne çizme ---
        static void DrawDetections(Mat frame, List<Detection> detections)
        {
            foreach (var det in detections)
            {
                Cv2.Rectangle(
                    frame,
                    new Point(det.X, det.Y),
                    new Point(det.X + det.Width, det.Y + det.Height),
                    Scalar.LimeGreen,
                    2
                );

                string label = $"{det.ClassName} {det.Confidence:P0} {det.DistanceM:F1}m";

                Cv2.PutText(
                    frame,
                    label,
                    new Point(det.X, Math.Max(20, det.Y - 5)),
                    HersheyFonts.HersheySimplex,
                    0.6,
                    Scalar.LimeGreen,
                    2
                );
            }
        }
    }

    class Detection
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }

        public string ClassName { get; set; } = "";
        public float Confidence { get; set; }

        public double DistanceM { get; set; }
        public double LateralM { get; set; }
    }
}