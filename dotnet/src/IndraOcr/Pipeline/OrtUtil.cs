using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace IndraOcr.Pipeline;

/// <summary>ONNX Runtime session options and tensor helpers (the Python constructor's settings).</summary>
public static class OrtUtil
{
    /// <summary>
    /// intra_op=threads, inter_op=1, all graph optimizations, no CPU memory arena
    /// (with changing line-batch shapes it kept growing in the Python package).
    /// </summary>
    public static SessionOptions Options(int threads)
    {
        var so = new SessionOptions
        {
            IntraOpNumThreads = threads,
            InterOpNumThreads = 1,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            EnableCpuMemArena = false,
        };
        return so;
    }

    public static DenseTensor<float> FloatTensor(float[] data, params int[] dims)
        => new(data, dims);

    public static DenseTensor<long> LongTensor(long[] data, params int[] dims)
        => new(data, dims);

    /// <summary>Flat values and shape of a float output that is not a scalar.</summary>
    public static (float[] data, int[] dims) Floats(DisposableNamedOnnxValue value)
    {
        var tensor = value.AsTensor<float>();
        return (tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    /// <summary>Sigmoid, exp and friends are used often enough to live here.</summary>
    public static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));

    public static void SoftmaxInPlace(float[] logits, int offset, int count)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < count; i++)
        {
            max = Math.Max(max, logits[offset + i]);
        }
        double sum = 0;
        for (int i = 0; i < count; i++)
        {
            double e = Math.Exp(logits[offset + i] - max);
            logits[offset + i] = (float)e;
            sum += e;
        }
        for (int i = 0; i < count; i++)
        {
            logits[offset + i] = (float)(logits[offset + i] / sum);
        }
    }
}
