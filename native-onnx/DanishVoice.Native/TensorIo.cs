using System.Text.Json;

namespace DanishVoice.Native;

/// <summary>
/// Reads the flat float32 .bin reference tensors (+ shapes.json) dumped by the
/// Python export scripts, and provides simple parity comparison helpers.
/// </summary>
internal static class TensorIo
{
    private static Dictionary<string, int[]>? shapes;

    public static void LoadShapes(string refsDir)
    {
        var json = File.ReadAllText(Path.Combine(refsDir, "shapes.json"));
        var raw = JsonSerializer.Deserialize<Dictionary<string, int[]>>(json)
                  ?? throw new InvalidOperationException("could not parse shapes.json");
        shapes = raw;
    }

    public static void AddShapes(string refsDir, string fileName)
    {
        shapes ??= [];
        var json = File.ReadAllText(Path.Combine(refsDir, fileName));
        var raw = JsonSerializer.Deserialize<Dictionary<string, int[]>>(json)
                  ?? throw new InvalidOperationException($"could not parse {fileName}");
        foreach (var kv in raw)
        {
            shapes[kv.Key] = kv.Value;
        }
    }

    public static int[] Shape(string name)
    {
        if (shapes is null)
        {
            throw new InvalidOperationException("call LoadShapes first");
        }
        return shapes[name];
    }

    public static float[] Load(string refsDir, string name)
    {
        var path = Path.Combine(refsDir, name + ".bin");
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    public static (double MaxAbs, double MeanAbs) Compare(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"length mismatch: {a.Length} vs {b.Length}");
        }
        double maxAbs = 0;
        double sumAbs = 0;
        for (var i = 0; i < a.Length; i++)
        {
            var d = Math.Abs((double)a[i] - b[i]);
            if (d > maxAbs)
            {
                maxAbs = d;
            }
            sumAbs += d;
        }
        return (maxAbs, sumAbs / a.Length);
    }

    public static void Report(string label, ReadOnlySpan<float> got, ReadOnlySpan<float> expected, double tol)
    {
        var (maxAbs, meanAbs) = Compare(got, expected);
        var ok = maxAbs <= tol;
        var status = ok ? "PASS" : "FAIL";
        Console.WriteLine($"  [{status}] {label}: maxAbs={maxAbs:E3} meanAbs={meanAbs:E3} (tol={tol:E1}, n={got.Length})");
    }
}
