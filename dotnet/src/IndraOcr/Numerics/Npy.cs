using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

namespace IndraOcr.Numerics;

/// <summary>One array from a .npy file: dtype, shape and the raw values.</summary>
public sealed class NpyArray
{
    public NpyArray(string descr, int[] shape, Array data)
    {
        Descr = descr;
        Shape = shape;
        Data = data;
    }

    public string Descr { get; }
    public int[] Shape { get; }
    public Array Data { get; }
    public int Length => Data.Length;

    public float[] AsFloat32() => Data switch
    {
        float[] a => a,
        double[] a => a.Select(v => (float)v).ToArray(),
        long[] a => a.Select(v => (float)v).ToArray(),
        ulong[] a => a.Select(v => (float)v).ToArray(),
        int[] a => a.Select(v => (float)v).ToArray(),
        byte[] a => a.Select(v => (float)v).ToArray(),
        _ => throw new InvalidOperationException($"cannot read {Descr} as float32"),
    };

    public double[] AsFloat64() => Data switch
    {
        double[] a => a,
        float[] a => a.Select(v => (double)v).ToArray(),
        _ => throw new InvalidOperationException($"cannot read {Descr} as float64"),
    };

    public long[] AsInt64() => Data switch
    {
        long[] a => a,
        int[] a => a.Select(v => (long)v).ToArray(),
        ulong[] a => a.Select(v => checked((long)v)).ToArray(),
        _ => throw new InvalidOperationException($"cannot read {Descr} as int64"),
    };

    public ulong[] AsUInt64() => Data switch
    {
        ulong[] a => a,
        long[] a => a.Select(v => unchecked((ulong)v)).ToArray(),
        _ => throw new InvalidOperationException($"cannot read {Descr} as uint64"),
    };

    public byte[] AsByte() => Data switch
    {
        byte[] a => a,
        _ => throw new InvalidOperationException($"cannot read {Descr} as byte"),
    };

    public bool[] AsBool() => Data switch
    {
        bool[] a => a,
        _ => throw new InvalidOperationException($"cannot read {Descr} as bool"),
    };
}

/// <summary>Reader for the .npy format (v1, v2 and v3), little endian.</summary>
public static class Npy
{
    private static readonly byte[] Magic = { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };

    public static NpyArray Read(string path)
    {
        using var s = File.OpenRead(path);
        return Read(s);
    }

    public static NpyArray Read(Stream s)
    {
        var magic = new byte[6];
        s.ReadExactly(magic);
        if (!magic.AsSpan().SequenceEqual(Magic))
        {
            throw new InvalidDataException("not a .npy file");
        }
        int major = s.ReadByte();
        int minor = s.ReadByte();
        int headerLen = major switch
        {
            1 => ReadUInt16(s),
            2 or 3 => checked((int)ReadUInt32(s)),
            _ => throw new InvalidDataException($"unsupported .npy version {major}.{minor}"),
        };
        var headerBytes = new byte[headerLen];
        s.ReadExactly(headerBytes);

        int dictStart = Array.IndexOf(headerBytes, (byte)'{');
        int dictEnd = Array.LastIndexOf(headerBytes, (byte)'}');
        if (dictStart < 0 || dictEnd < dictStart)
        {
            throw new InvalidDataException("bad .npy header");
        }
        string header = Encoding.UTF8.GetString(headerBytes, dictStart, dictEnd - dictStart + 1);

        var descr = Regex.Match(header, @"'descr'\s*:\s*'([^']+)'").Groups[1].Value;
        bool fortran = Regex.Match(header, @"'fortran_order'\s*:\s*(True|False)").Groups[1].Value == "True";
        string shapeText = Regex.Match(header, @"'shape'\s*:\s*\(([^)]*)\)").Groups[1].Value;
        var shape = shapeText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(v => int.Parse(v))
            .ToArray();
        if (fortran)
        {
            throw new NotSupportedException("Fortran-ordered .npy is not supported");
        }
        if (descr.Length > 0 && descr[0] == '>')
        {
            throw new NotSupportedException("big-endian .npy is not supported");
        }

        long count = 1;
        foreach (int d in shape)
        {
            count *= d;
        }
        if (shape.Length == 0)
        {
            count = 1;
        }

        var data = ReadData(s, descr, checked((int)count));
        return new NpyArray(descr, shape, data);
    }

    private static ushort ReadUInt16(Stream s)
    {
        Span<byte> b = stackalloc byte[2];
        s.ReadExactly(b);
        return BinaryPrimitives.ReadUInt16LittleEndian(b);
    }

    private static uint ReadUInt32(Stream s)
    {
        Span<byte> b = stackalloc byte[4];
        s.ReadExactly(b);
        return BinaryPrimitives.ReadUInt32LittleEndian(b);
    }

    private static Array ReadData(Stream s, string descr, int count)
    {
        var bytes = new byte[checked(count * DTypeSize(descr))];
        s.ReadExactly(bytes);
        switch (descr)
        {
            case "<f4":
            {
                var a = new float[count];
                for (int i = 0; i < count; i++)
                {
                    a[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
                }
                return a;
            }
            case "<f8":
            {
                var a = new double[count];
                for (int i = 0; i < count; i++)
                {
                    a[i] = BinaryPrimitives.ReadDoubleLittleEndian(bytes.AsSpan(i * 8, 8));
                }
                return a;
            }
            case "<i8":
            {
                var a = new long[count];
                for (int i = 0; i < count; i++)
                {
                    a[i] = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(i * 8, 8));
                }
                return a;
            }
            case "<u8":
            {
                var a = new ulong[count];
                for (int i = 0; i < count; i++)
                {
                    a[i] = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(i * 8, 8));
                }
                return a;
            }
            case "<i4":
            {
                var a = new int[count];
                for (int i = 0; i < count; i++)
                {
                    a[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * 4, 4));
                }
                return a;
            }
            case "<i2":
            {
                var a = new short[count];
                for (int i = 0; i < count; i++)
                {
                    a[i] = BinaryPrimitives.ReadInt16LittleEndian(bytes.AsSpan(i * 2, 2));
                }
                return a;
            }
            case "<u2":
            {
                var a = new ushort[count];
                for (int i = 0; i < count; i++)
                {
                    a[i] = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2, 2));
                }
                return a;
            }
            case "|u1":
            case "<u1":
            case "|i1":
            case "<i1":
                return bytes;
            case "|b1":
            case "<b1":
            {
                var a = new bool[count];
                for (int i = 0; i < count; i++)
                {
                    a[i] = bytes[i] != 0;
                }
                return a;
            }
            default:
                throw new NotSupportedException($"unsupported .npy dtype {descr}");
        }
    }

    private static int DTypeSize(string descr) => descr[^1] switch
    {
        '1' when descr.StartsWith("|b") || descr.Contains('u') || descr.Contains('i') => 1,
        '1' => 1,
        '2' => 2,
        '4' => 4,
        '8' => 8,
        _ => throw new NotSupportedException($"unsupported .npy dtype {descr}"),
    };
}

/// <summary>Reader for .npz: a ZIP of .npy entries (numpy.savez).</summary>
public static class Npz
{
    public static Dictionary<string, NpyArray> ReadAll(string path)
    {
        var outv = new Dictionary<string, NpyArray>(StringComparer.Ordinal);
        using var zip = ZipFile.OpenRead(path);
        foreach (var entry in zip.Entries)
        {
            if (!entry.Name.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            using var s = entry.Open();
            outv[Path.GetFileNameWithoutExtension(entry.Name)] = Npy.Read(s);
        }
        return outv;
    }
}
