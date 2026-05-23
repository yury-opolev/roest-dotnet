namespace DanishVoice.Native.Dsp;

/// <summary>
/// STFT / iSTFT that replicate torch.stft / torch.istft for the HiFT vocoder
/// configuration: real input, center=True, pad_mode='reflect', window=hann
/// (periodic), onesided=True, normalized=False.
///
/// n_fft is tiny (16), so a direct O(n_fft^2) DFT is used for clarity.
/// </summary>
internal sealed class ShortTimeFourier
{
    private readonly int nFft;
    private readonly int hop;
    private readonly int nBins;          // n_fft/2 + 1
    private readonly float[] window;     // length n_fft

    public ShortTimeFourier(int nFft, int hop, float[] window)
    {
        this.nFft = nFft;
        this.hop = hop;
        this.window = window;
        this.nBins = nFft / 2 + 1;
        if (window.Length != nFft)
        {
            throw new ArgumentException("window length must equal n_fft");
        }
    }

    /// <summary>Hann periodic window (matches scipy get_window("hann", n, fftbins=True)).</summary>
    public static float[] HannPeriodic(int n)
    {
        var w = new float[n];
        for (var i = 0; i < n; i++)
        {
            w[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / n));
        }
        return w;
    }

    /// <summary>
    /// Forward STFT of a single real signal. Returns interleaved [real(nBins,frames);
    /// imag(nBins,frames)] flattened row-major to match the Python s_stft layout
    /// (concat of real and imag along the channel axis).
    /// </summary>
    public (float[] Real, float[] Imag, int Frames) Forward(ReadOnlySpan<float> signal)
    {
        var pad = this.nFft / 2;
        var padded = ReflectPad(signal, pad);
        var frames = 1 + (padded.Length - this.nFft) / this.hop;

        var real = new float[this.nBins * frames];
        var imag = new float[this.nBins * frames];

        var frame = new float[this.nFft];
        for (var m = 0; m < frames; m++)
        {
            var start = m * this.hop;
            for (var n = 0; n < this.nFft; n++)
            {
                frame[n] = padded[start + n] * this.window[n];
            }
            // onesided DFT: bins 0..nBins-1
            for (var k = 0; k < this.nBins; k++)
            {
                double re = 0, im = 0;
                var w = -2.0 * Math.PI * k / this.nFft;
                for (var n = 0; n < this.nFft; n++)
                {
                    var ang = w * n;
                    re += frame[n] * Math.Cos(ang);
                    im += frame[n] * Math.Sin(ang);
                }
                real[k * frames + m] = (float)re;
                imag[k * frames + m] = (float)im;
            }
        }
        return (real, imag, frames);
    }

    /// <summary>
    /// Inverse STFT. magnitude/phase are [nBins, frames] row-major; the complex
    /// spectrum is mag*exp(i*phase). Returns the reconstructed real signal with
    /// center padding trimmed (length = hop*(frames-1) for center=True ... matched
    /// to torch's inferred length).
    /// </summary>
    public float[] Inverse(ReadOnlySpan<float> magnitude, ReadOnlySpan<float> phase, int frames)
    {
        var pad = this.nFft / 2;
        var fullLen = this.nFft + this.hop * (frames - 1);
        var y = new double[fullLen];
        var env = new double[fullLen];

        var spec = new (double Re, double Im)[this.nFft];
        var frameTime = new double[this.nFft];

        for (var m = 0; m < frames; m++)
        {
            // rebuild full Hermitian spectrum from onesided bins
            for (var k = 0; k < this.nBins; k++)
            {
                var mag = Math.Min(magnitude[k * frames + m], 1e2); // matches torch.clip(max=1e2)
                var ph = phase[k * frames + m];
                spec[k] = (mag * Math.Cos(ph), mag * Math.Sin(ph));
            }
            for (var k = this.nBins; k < this.nFft; k++)
            {
                var conj = spec[this.nFft - k];
                spec[k] = (conj.Re, -conj.Im);
            }
            // IDFT -> real frame
            for (var n = 0; n < this.nFft; n++)
            {
                double re = 0;
                var w = 2.0 * Math.PI * n / this.nFft;
                for (var k = 0; k < this.nFft; k++)
                {
                    var ang = w * k;
                    re += spec[k].Re * Math.Cos(ang) - spec[k].Im * Math.Sin(ang);
                }
                frameTime[n] = re / this.nFft;
            }
            // overlap-add with window, accumulate window^2 envelope
            var start = m * this.hop;
            for (var n = 0; n < this.nFft; n++)
            {
                y[start + n] += frameTime[n] * this.window[n];
                env[start + n] += this.window[n] * this.window[n];
            }
        }

        // normalize by window envelope (NOLA), then trim center padding
        var outLen = fullLen - 2 * pad;
        var outSig = new float[outLen];
        for (var i = 0; i < outLen; i++)
        {
            var idx = i + pad;
            var e = env[idx];
            outSig[i] = (float)(e > 1e-11 ? y[idx] / e : 0.0);
        }
        return outSig;
    }

    private static float[] ReflectPad(ReadOnlySpan<float> x, int pad)
    {
        var n = x.Length;
        var outLen = n + 2 * pad;
        var r = new float[outLen];
        for (var i = 0; i < n; i++)
        {
            r[pad + i] = x[i];
        }
        // reflect (no edge repeat): torch reflect uses x[pad], x[pad-1]... mirror about edge
        for (var i = 0; i < pad; i++)
        {
            r[pad - 1 - i] = x[i + 1];
            r[pad + n + i] = x[n - 2 - i];
        }
        return r;
    }
}
