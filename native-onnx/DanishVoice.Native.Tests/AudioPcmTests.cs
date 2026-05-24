namespace DanishVoice.Native.Tests;

public class AudioPcmTests
{
    [Fact]
    public void FloatToPcm16_ProducesTwoBytesPerSample()
    {
        var bytes = AudioPcm.FloatToPcm16(new[] { 0f, 0f, 0f });
        Assert.Equal(6, bytes.Length);
    }

    [Fact]
    public void FloatToPcm16_EmptyInputProducesEmptyOutput()
    {
        var bytes = AudioPcm.FloatToPcm16(ReadOnlySpan<float>.Empty);
        Assert.Empty(bytes);
    }

    [Fact]
    public void FloatToPcm16_ZeroIsZeroLittleEndian()
    {
        var bytes = AudioPcm.FloatToPcm16(new[] { 0f });
        Assert.Equal(new byte[] { 0x00, 0x00 }, bytes);
    }

    [Fact]
    public void FloatToPcm16_ClampsAboveOneToMaxShort()
    {
        var bytes = AudioPcm.FloatToPcm16(new[] { 2.0f });
        Assert.Equal(new byte[] { 0xFF, 0x7F }, bytes); // 32767 = 0x7FFF LE
    }

    [Fact]
    public void FloatToPcm16_ClampsBelowMinusOneToMinShort()
    {
        var bytes = AudioPcm.FloatToPcm16(new[] { -2.0f });
        Assert.Equal(new byte[] { 0x00, 0x80 }, bytes); // -32768 = 0x8000 LE
    }
}
