namespace DanishVoice.Native;

/// <summary>Writes mono float samples as a 16-bit PCM WAV file.</summary>
public static class WavWriter
{
    public static void Write(string path, float[] samples, int sampleRate)
    {
        using var fs = new FileStream(path, FileMode.Create);
        using var bw = new BinaryWriter(fs);
        var n = samples.Length;
        var byteRate = sampleRate * 2;
        bw.Write("RIFF"u8.ToArray());
        bw.Write(36 + n * 2);
        bw.Write("WAVE"u8.ToArray());
        bw.Write("fmt "u8.ToArray());
        bw.Write(16);
        bw.Write((short)1);          // PCM
        bw.Write((short)1);          // mono
        bw.Write(sampleRate);
        bw.Write(byteRate);
        bw.Write((short)2);          // block align
        bw.Write((short)16);         // bits per sample
        bw.Write("data"u8.ToArray());
        bw.Write(n * 2);
        foreach (var s in samples)
        {
            bw.Write((short)Math.Clamp(s * 32767f, -32768f, 32767f));
        }
    }
}
