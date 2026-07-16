using System;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Threading;

namespace MyAvaloniaApp.Views;

public partial class MainWindow : Window
{
    private TcpListener? _server;
    private CancellationTokenSource? _cts;

    private const double CanvasSize = 320.0;

    // Haritanın ölçeği
    private const double MapForwardM = 8.0;   // ileri yönde 8 metre göster
    private const double MapHalfWidthM = 4.0; // sağ-sol 4'er metre göster

    public MainWindow()
    {
        InitializeComponent();
    }

    private void StartButton_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            Console.WriteLine("START BUTTON BASILDI");

            if (_server != null)
            {
                StatusText.Text = "Server zaten çalışıyor.";
                return;
            }

            _cts = new CancellationTokenSource();

            _server = new TcpListener(IPAddress.Any, 5055);
            _server.Start();

            Console.WriteLine("SERVER STARTED");

            StatusText.Text = "TCP Server çalışıyor. Port:5055";

            _ = Task.Run(() => AcceptLoop(_cts.Token));
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.ToString());
            StatusText.Text = ex.Message;
        }
    }
    private async Task AcceptLoop(CancellationToken token)
    {
        try
        {
            while (!token.IsCancellationRequested && _server != null)
            {
                TcpClient client = await _server.AcceptTcpClientAsync(token);

                Dispatcher.UIThread.Post(() =>
                {
                    StatusText.Text = "Görüntü işleme bağlantısı geldi.";
                    AppendLog("CLIENT CONNECTED");
                });

                _ = Task.Run(() => ReadClientLoop(client, token), token);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            Dispatcher.UIThread.Post(() =>
            {
                StatusText.Text = "Server hatası: " + ex.Message;
                AppendLog("SERVER ERROR: " + ex.Message);
            });
        }
    }

    private async Task ReadClientLoop(TcpClient client, CancellationToken token)
    {
        try
        {
            using (client)
            using (NetworkStream stream = client.GetStream())
            using (StreamReader reader = new StreamReader(stream, Encoding.UTF8))
            {
                while (!token.IsCancellationRequested)
                {
                    string? line = await reader.ReadLineAsync(token);

                    if (line == null)
                        break;

                    Dispatcher.UIThread.Post(() =>
                    {
                        AppendLog(line);
                        ParseFrameLine(line);
                    });
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            Dispatcher.UIThread.Post(() =>
            {
                AppendLog("CLIENT ERROR: " + ex.Message);
            });
        }
    }

    private void ParseFrameLine(string line)
    {
        // Beklenen format:
        // FRAME,640,480;yellow,0.91,250,130,80,120,2.40,-0.30
        //
        // FRAME,frameW,frameH;color,conf,x,y,w,h,distanceM,lateralM

        if (!line.StartsWith("FRAME,", StringComparison.OrdinalIgnoreCase))
            return;

        VisionCanvas.Children.Clear();
        DrawBoat();
        DrawGrid();

        string[] parts = line.Split(';', StringSplitOptions.RemoveEmptyEntries);

        if (parts.Length == 0)
            return;

        int detectionCount = 0;

        for (int i = 1; i < parts.Length; i++)
        {
            string[] f = parts[i].Split(',');

            if (f.Length < 8)
                continue;

            string colorName = f[0];

            float.TryParse(f[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float conf);

            // x,y,w,h şimdilik log için duruyor; haritada distance/lateral kullanıyoruz.
            int.TryParse(f[2], out int x);
            int.TryParse(f[3], out int y);
            int.TryParse(f[4], out int w);
            int.TryParse(f[5], out int h);

            double.TryParse(f[6], NumberStyles.Float, CultureInfo.InvariantCulture, out double distanceM);
            double.TryParse(f[7], NumberStyles.Float, CultureInfo.InvariantCulture, out double lateralM);

            DrawMapDetection(colorName, conf, distanceM, lateralM);
            detectionCount++;
        }

        StatusText.Text = detectionCount == 0
            ? "FRAME geldi ama tespit yok."
            : $"{detectionCount} tespit haritaya çizildi.";
    }

    private void DrawBoat()
    {
        double boatX = CanvasSize / 2.0;
        double boatY = CanvasSize - 25;

        var boat = new Polygon
        {
            Points =
            {
                new Point(boatX, boatY - 18),
                new Point(boatX - 12, boatY + 12),
                new Point(boatX + 12, boatY + 12)
            },
            Fill = Brushes.DeepSkyBlue,
            Stroke = Brushes.White,
            StrokeThickness = 1
        };

        VisionCanvas.Children.Add(boat);

        var label = new TextBlock
        {
            Text = "İDA",
            Foreground = Brushes.White,
            FontSize = 12,
            FontWeight = FontWeight.Bold
        };

        Canvas.SetLeft(label, boatX - 12);
        Canvas.SetTop(label, boatY + 14);

        VisionCanvas.Children.Add(label);
    }

    private void DrawGrid()
    {
        // Orta çizgi
        var centerLine = new Line
        {
            StartPoint = new Point(CanvasSize / 2, 10),
            EndPoint = new Point(CanvasSize / 2, CanvasSize - 45),
            Stroke = Brushes.DarkSlateGray,
            StrokeThickness = 1
        };

        VisionCanvas.Children.Add(centerLine);

        // 2m, 4m, 6m, 8m mesafe çizgileri
        for (int m = 2; m <= 8; m += 2)
        {
            double usableHeight = CanvasSize - 60;
            double y = CanvasSize - 45 - (m / MapForwardM) * usableHeight;

            var line = new Line
            {
                StartPoint = new Point(10, y),
                EndPoint = new Point(CanvasSize - 10, y),
                Stroke = Brushes.DarkSlateGray,
                StrokeThickness = 1
            };

            VisionCanvas.Children.Add(line);

            var label = new TextBlock
            {
                Text = $"{m}m",
                Foreground = Brushes.Gray,
                FontSize = 10
            };

            Canvas.SetLeft(label, 12);
            Canvas.SetTop(label, y - 12);

            VisionCanvas.Children.Add(label);
        }
    }

    private void DrawMapDetection(string colorName, float confidence, double distanceM, double lateralM)
    {
        if (distanceM <= 0)
            return;

        double usableHeight = CanvasSize - 60;
        double usableWidth = CanvasSize - 40;

        double mapX = CanvasSize / 2.0 + (lateralM / MapHalfWidthM) * (usableWidth / 2.0);
        double mapY = CanvasSize - 45 - (distanceM / MapForwardM) * usableHeight;

        mapX = Math.Clamp(mapX, 10, CanvasSize - 10);
        mapY = Math.Clamp(mapY, 10, CanvasSize - 45);

        IBrush brush = GetBrush(colorName);

        double size = colorName.ToLowerInvariant() switch
        {
            "black" => 22,
            "red" => 22,
            "green" => 22,
            "yellow" => 16,
            "orange" => 16,
            _ => 16
        };

        var obstacle = new Ellipse
        {
            Width = size,
            Height = size,
            Fill = brush,
            Stroke = Brushes.White,
            StrokeThickness = 1
        };

        Canvas.SetLeft(obstacle, mapX - size / 2.0);
        Canvas.SetTop(obstacle, mapY - size / 2.0);

        VisionCanvas.Children.Add(obstacle);

        var label = new TextBlock
        {
            Text = $"{colorName}\n{distanceM:F1} m",
            Foreground = Brushes.White,
            FontSize = 11,
            FontWeight = FontWeight.Bold
        };

        Canvas.SetLeft(label, mapX + 8);
        Canvas.SetTop(label, mapY - 12);

        VisionCanvas.Children.Add(label);
    }

    private static IBrush GetBrush(string colorName)
    {
        return colorName.ToLowerInvariant() switch
        {
            "green" => Brushes.LimeGreen,
            "red" => Brushes.Red,
            "yellow" => Brushes.Yellow,
            "orange" => Brushes.Orange,
            "black" => Brushes.Gray,
            _ => Brushes.DeepSkyBlue
        };
    }

    private void AppendLog(string text)
    {
        string time = DateTime.Now.ToString("HH:mm:ss");

        LogBox.Text = $"[{time}] {text}\n" + LogBox.Text;

        if (LogBox.Text.Length > 8000)
            LogBox.Text = LogBox.Text.Substring(0, 8000);
    }

    private void ClearButton_Click(object? sender, RoutedEventArgs e)
    {
        VisionCanvas.Children.Clear();
        LogBox.Text = "";
        StatusText.Text = "Temizlendi.";
    }

    protected override void OnClosed(EventArgs e)
    {
        _cts?.Cancel();
        _server?.Stop();

        base.OnClosed(e);
    }
}