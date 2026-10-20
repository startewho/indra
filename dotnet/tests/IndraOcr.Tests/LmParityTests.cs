using System.Text.Json;
using System.Text.Json.Serialization;
using IndraOcr.Lm;
using IndraOcr.Numerics;

namespace IndraOcr.Tests;

/// <summary>S2/S5 parity: the n-gram LM and the CTC beam search (tools/dump_lm_fixture.py).</summary>
public class LmParityTests
{
    private sealed record Probe(int[] History, int Token, double Logp);

    private sealed record BeamCase(int[] Ids, int[] Steps);

    private sealed record Fixture(
        int Order,
        [property: JsonPropertyName("vocab_size")] int VocabSize,
        int Bits,
        [property: JsonPropertyName("max_logp")] double MaxLogp,
        double Alpha, double Beta,
        [property: JsonPropertyName("beam_size")] int BeamSize,
        [property: JsonPropertyName("cand_floor")] double CandFloor,
        Probe[] Probes, BeamCase Beam);

    [SkippableFact]
    public void LmAndBeamSearchMatchPython()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        string dir = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "lm");
        string fixturePath = Path.Combine(dir, "fixture.json");
        Skip.IfNot(File.Exists(fixturePath), "no LM fixture (run tools/dump_lm_fixture.py)");

        var fixture = JsonSerializer.Deserialize<Fixture>(
            File.ReadAllText(fixturePath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        var lm = NgramLm.Load(Path.Combine(TestPaths.Models!, "lm.npz"));

        Assert.Equal(fixture.Order, lm.Order);
        Assert.Equal(fixture.VocabSize, lm.VocabSize);
        Assert.Equal(fixture.Bits, lm.Bits);
        Assert.True(Math.Abs(fixture.MaxLogp - lm.MaxLogp) <= 1e-9,
            $"max_logp: expected {fixture.MaxLogp}, got {lm.MaxLogp}");

        foreach (var probe in fixture.Probes)
        {
            double got = lm.Logp(probe.History, probe.Token);
            Assert.True(Math.Abs(probe.Logp - got) <= 1e-5,
                $"logp({string.Join(',', probe.History)}, {probe.Token}): " +
                $"expected {probe.Logp}, got {got}");
        }

        var blank = Npy.Read(Path.Combine(dir, "blank.npy")).AsFloat32();
        var topvArr = Npy.Read(Path.Combine(dir, "topv.npy"));
        var topi = Npy.Read(Path.Combine(dir, "topi.npy")).AsInt64();
        var (ids, steps) = BeamSearch.Search(
            blank, topvArr.AsFloat32(), topi, blank.Length, topvArr.Shape[1],
            lm, fixture.Alpha, fixture.Beta, fixture.BeamSize, candFloor: fixture.CandFloor);
        Assert.True(fixture.Beam.Ids.SequenceEqual(ids),
            $"beam ids: expected [{string.Join(',', fixture.Beam.Ids)}], got [{string.Join(',', ids)}]");
        Assert.True(fixture.Beam.Steps.SequenceEqual(steps),
            $"beam steps: expected [{string.Join(',', fixture.Beam.Steps)}], got [{string.Join(',', steps)}]");
    }
}
