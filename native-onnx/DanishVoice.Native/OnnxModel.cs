using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DanishVoice.Native;

/// <summary>Thin wrapper around an ONNX Runtime session for float in/out graphs.</summary>
internal sealed class OnnxModel : IDisposable
{
    private readonly InferenceSession session;

    public OnnxModel(string path, bool useCuda = false)
    {
        if (useCuda)
        {
            using var opts = SessionOptions.MakeSessionOptionWithCudaProvider(0);
            this.session = new InferenceSession(path, opts);
        }
        else
        {
            this.session = new InferenceSession(path);
        }
    }

    public IReadOnlyList<string> InputNames => [.. this.session.InputMetadata.Keys];

    public IReadOnlyList<string> OutputNames => [.. this.session.OutputMetadata.Keys];

    public Dictionary<string, (float[] Data, int[] Dims)> Run(
        IReadOnlyDictionary<string, (float[] Data, int[] Dims)> inputs,
        IReadOnlyDictionary<string, (long[] Data, int[] Dims)>? longInputs = null)
    {
        var feeds = new List<NamedOnnxValue>(inputs.Count);
        foreach (var (name, value) in inputs)
        {
            var tensor = new DenseTensor<float>(value.Data, value.Dims);
            feeds.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
        }
        if (longInputs is not null)
        {
            foreach (var (name, value) in longInputs)
            {
                var tensor = new DenseTensor<long>(value.Data, value.Dims);
                feeds.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
            }
        }

        using var results = this.session.Run(feeds);
        var outputs = new Dictionary<string, (float[], int[])>();
        foreach (var r in results)
        {
            var t = r.AsTensor<float>();
            outputs[r.Name] = (t.ToArray(), [.. t.Dimensions]);
        }
        return outputs;
    }

    public void Dispose()
    {
        this.session.Dispose();
    }
}
