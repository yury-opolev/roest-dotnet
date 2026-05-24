using System.Buffers.Binary;

namespace DanishVoice.Native;

/// <summary>Conversions between float samples and 16-bit PCM bytes.</summary>
internal static class AudioPcm
{
    /// <summary>
    /// Convert mono float samples in [-1, 1] to little-endian 16-bit PCM bytes.
    /// Output is always 2 bytes per sample (sample-aligned). Matches the int16
    /// quantization previously inlined in <see cref="WavWriter"/>.
    /// </summary>
    public static byte[] FloatToPcm16(ReadOnlySpan<float> samples)
    {
        var result = new byte[samples.Length * 2];
        for (var i = 0; i < samples.Length; i++)
        {
            var value = (short)Math.Clamp(samples[i] * 32767f, -32768f, 32767f);
            BinaryPrimitives.WriteInt16LittleEndian(result.AsSpan(i * 2), value);
        }
        return result;
    }
}
