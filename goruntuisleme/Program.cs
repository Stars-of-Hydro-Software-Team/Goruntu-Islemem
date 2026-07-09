using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace IdaHavuzTesti
{
    class Program
    {
        // train.ipynb'deki model.names sırasına göre (Roboflow "ida-wuieq" v1 dataseti):
        // black, green, orange, red, yellow
        static readonly string[] ClassNames = { "black", "green", "orange", "red", "yellow" };

        const int InputSize = 640;          // Modelin eğitildiği giriş boyutu
        const float ConfThreshold = 0.5f;   // Şartname: %50 güven eşiği
        const float NmsThreshold = 0.45f;   // Çakışan kutuları elemek için IoU eşiği

        static void Main(string[] args)
        {
            Console.WriteLine("Havuz Testi Baslatiliyor...");

            // 1. Modelin Yuklenmesi
            string modelPath = "best.onnx";
            using var session = new InferenceSession(modelPath);

            // 2. Kamera Baglantisi
            using var capture = new VideoCapture(0);
            if (!capture.IsOpened())
            {
                Console.WriteLine("HATA: Kamera acilamadi. Baglantiyi kontrol edin.");
                return;
            }
            capture.Set(VideoCaptureProperties.FrameWidth, 640);
            capture.Set(VideoCaptureProperties.FrameHeight, 480);

            // 3. Video Kaydedici
            var fourcc = VideoWriter.FourCC('m', 'p', '4', 'v');
            using var writer = new VideoWriter("havuz_testi_kayit.mp4", fourcc, 10, new Size(640, 480));

            using var frame = new Mat();

            try
            {
                while (true)
                {
                    capture.Read(frame);
                    if (frame.Empty()) break;

                    // 4. On Isleme: Gama duzeltmesi
                    using Mat processedFrame = ApplyGammaCorrection(frame, 1.2);

                    // 5. YOLOv8 ONNX Cikarimi
                    var detections = RunYoloInference(session, processedFrame);

                    // Terminale görülen renkleri yazdır
                    if (detections.Count > 0)
                    {
                        foreach (var det in detections)
                        {
                        Console.WriteLine(
                            $"DETECT,{det.ClassName},{det.Confidence:F2},{det.X},{det.Y},{det.Width},{det.Height}"
                        );
                        Console.Out.Flush();
                        }
                    }


                    // 6. Tespitleri Ekrana Ciz
                    DrawDetections(processedFrame, detections);

                    // 7. Goster ve Kaydet
                    writer.Write(processedFrame);
                    Cv2.ImShow("IDA Havuz Testi Gorusu", processedFrame);

                    if (Cv2.WaitKey(1) == 'q') break;
                }
            }
            finally
            {
                // Kaynaklarin duzgun serbest birakildigindan emin ol
                capture.Release();
                writer.Release();
                Cv2.DestroyAllWindows();
            }
        }

        // --- Gama Duzeltmesi ---
        static Mat ApplyGammaCorrection(Mat original, double gamma)
        {
            using Mat floatImage = new Mat();
            original.ConvertTo(floatImage, MatType.CV_32F, 1.0 / 255.0);
            Cv2.Pow(floatImage, gamma, floatImage);

            Mat result = new Mat();
            floatImage.ConvertTo(result, MatType.CV_8U, 255.0);
            return result;
        }

        // --- YOLOv8 ONNX Cikarimi ---
        static List<Detection> RunYoloInference(InferenceSession session, Mat frame)
        {
            int origW = frame.Width;
            int origH = frame.Height;

            // --- 5a. Letterbox: en/boy oranini bozmadan 640x640'a yerlestir ---
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

            // --- 5b. BGR -> RGB, HWC -> CHW, normalize [0,1] ---
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

            // --- 5c. Cikarimi calistir ---
            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>(); // Beklenen sekil: [1, 4+numClasses, N]

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

                if (bestScore < ConfThreshold) continue;

                float x1 = cx - w / 2f;
                float y1 = cy - h / 2f;

                boxes.Add(new Rect2d(x1, y1, w, h));
                scores.Add(bestScore);
                classIds.Add(bestClass);
            }

            // --- 5d. NMS ile cakisan kutulari ele ---
            CvDnn.NMSBoxes(boxes, scores, ConfThreshold, NmsThreshold, out int[] keepIndices);

            var detections = new List<Detection>();
            foreach (int idx in keepIndices)
            {
                var box = boxes[idx];

                // Letterbox paddingini cikar ve orijinal frame boyutuna geri olcekle
                float x1 = (float)((box.X - padX) / scale);
                float y1 = (float)((box.Y - padY) / scale);
                float w = (float)(box.Width / scale);
                float h = (float)(box.Height / scale);

                // Sinirlarin frame disina tasmasini engelle
                x1 = Math.Clamp(x1, 0, origW - 1);
                y1 = Math.Clamp(y1, 0, origH - 1);
                w = Math.Clamp(w, 0, origW - x1);
                h = Math.Clamp(h, 0, origH - y1);

                int classId = classIds[idx];
                string className = (classId >= 0 && classId < ClassNames.Length)
                    ? ClassNames[classId]
                    : $"Sinif_{classId}";

                detections.Add(new Detection
                {
                    X = (int)x1,
                    Y = (int)y1,
                    Width = (int)w,
                    Height = (int)h,
                    ClassName = className,
                    Confidence = scores[idx]
                });
            }

            return detections;
        }

        // --- Tespitleri Cizme ---
        static void DrawDetections(Mat frame, List<Detection> detections)
        {
            foreach (var det in detections)
            {
                Cv2.Rectangle(frame, new Point(det.X, det.Y), new Point(det.X + det.Width, det.Y + det.Height), Scalar.LimeGreen, 2);

                string label = $"{det.ClassName} {det.Confidence:P0}";
                Cv2.PutText(frame, label, new Point(det.X, det.Y - 5), HersheyFonts.HersheySimplex, 0.6, Scalar.LimeGreen, 2);
            }
        }
    }

    class Detection
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public string ClassName { get; set; }
        public float Confidence { get; set; }
    }
}
