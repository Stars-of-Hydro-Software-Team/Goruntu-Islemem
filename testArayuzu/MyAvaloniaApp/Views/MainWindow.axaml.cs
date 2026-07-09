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

    public MainWindow()
    {
        InitializeComponent();
    }

    private void StartButton_Click(object? sender, RoutedEventArgs e)
    {
        if (_server != null)
        {
            StatusText.Text = "Server zaten çalışıyor: 127.0.0.1:5055";
            return;
        }

        _cts = new CancellationTokenSource();
        _server = new TcpListener(IPAddress.Loopback, 5055);
        _server.Start();

        StatusText.Text = "TCP Server çalışıyor: 127.0.0.1:5055";

        _ = Task.Run(() => AcceptLoop(_cts.Token));
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
        // FRAME,640,480;green,0.91,80,120,90,100;red,0.88,420,180,80,90

        if (!line.StartsWith("FRAME,", StringComparison.OrdinalIgnoreCase))
            return;

        VisionCanvas.Children.Clear();

        string[] parts = line.Split(';', StringSplitOptions.RemoveEmptyEntries);

        if (parts.Length == 0)
            return;

        string[] header = parts[0].Split(',');

        int frameW = 640;
        int frameH = 480;

        if (header.Length >= 3)
        {
            int.TryParse(header[1], out frameW);
            int.TryParse(header[2], out frameH);
        }

        if (frameW <= 0) frameW = 640;
        if (frameH <= 0) frameH = 480;

        int detectionCount = 0;

        for (int i = 1; i < parts.Length; i++)
        {
            string[] f = parts[i].Split(',');

            if (f.Length < 6)
                continue;

            string colorName = f[0];

            float.TryParse(f[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float conf);
            int.TryParse(f[2], out int x);
            int.TryParse(f[3], out int y);
            int.TryParse(f[4], out int w);
            int.TryParse(f[5], out int h);

            DrawDetection(colorName, conf, x, y, w, h, frameW, frameH);
            detectionCount++;
        }

        StatusText.Text = detectionCount == 0
            ? "FRAME geldi ama tespit yok."
            : $"{detectionCount} tespit çizildi.";
    }

    private void DrawDetection(string colorName, float confidence, int x, int y, int w, int h, int frameW, int frameH)
    {
        double drawX = Math.Clamp((double)x / frameW * CanvasSize, 0, CanvasSize - 5);
        double drawY = Math.Clamp((double)y / frameH * CanvasSize, 0, CanvasSize - 5);
        double drawW = Math.Clamp((double)w / frameW * CanvasSize, 14, CanvasSize - drawX);
        double drawH = Math.Clamp((double)h / frameH * CanvasSize, 14, CanvasSize - drawY);

        IBrush brush = GetBrush(colorName);

        var rect = new Rectangle
        {
            Width = drawW,
            Height = drawH,
            Stroke = brush,
            StrokeThickness = 3,
            Fill = Brushes.Transparent
        };

        Canvas.SetLeft(rect, drawX);
        Canvas.SetTop(rect, drawY);
        VisionCanvas.Children.Add(rect);

        var label = new TextBlock
        {
            Text = $"{colorName} {(confidence * 100):F0}%",
            Foreground = brush,
            FontSize = 13,
            FontWeight = FontWeight.Bold
        };

        Canvas.SetLeft(label, drawX);
        Canvas.SetTop(label, Math.Max(0, drawY - 20));
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