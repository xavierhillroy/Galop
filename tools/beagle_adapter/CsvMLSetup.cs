using BeagleLib.Engine;
using BeagleLib.VM;

namespace Run.MLSetups;

// =============================================================================
// GALOP-campaign adapter (harness-level; the Beagle evolutionary engine is
// untouched). Reads one of OUR benchmark CSVs (last column = target) and
// serves it round-robin, so with ExperimentsPerGeneration == row count every
// generation evaluates the full dataset (full-batch parity with GALOP/EvoGP;
// only possible for datasets with <= 1024 rows — Beagle's one-block-per-
// individual design cap). Configured by environment variables:
//   BEAGLE_CSV     path to training CSV (header row; last column = target)
//   BEAGLE_COLONY  fixed target colony size (population parity; Beagle's own
//                  dynamics give a -20%/+60% band around it — their "min 80%,
//                  max 160%" comment, measured 59-152M on a 100M target)
//   BEAGLE_EPG     fitness cases per generation (default: row count rounded
//                  down to a multiple of 32, capped at 1024 — both per the
//                  authors' tutorial (arXiv 2607.06731 s3.7: "optimal
//                  performance requires values divisible by 32"; 1024 = the
//                  documented block limit). The engine ACCEPTS any value
//                  (verified MLEngine.cs:130,839-840); we follow their
//                  documented guidance, matching the recorded campaign runs.
// Parity choices: primitive set restricted to {Add,Sub,Mul,Div,Sin,Cos} plus
// Beagle's structural commands (Const/Load/Dup/Del/Swap/Copy/Paste — these
// are representation machinery, like registers, not math primitives); colony
// resets disabled; run does not stop on perfect ASR (we cut at generation
// 100 from the log). Everything else = Beagle as shipped.
// =============================================================================
public class CsvMLSetup : MLSetup
{
    private readonly string[] _labels;
    private readonly float[][] _inputs;
    private readonly float[] _outputs;
    private readonly int _colony;
    private uint _epg;   // lazily computed — see ExperimentsPerGeneration
    private long _cursor = -1;

    public CsvMLSetup()
    {
        var path = Environment.GetEnvironmentVariable("BEAGLE_CSV")
                   ?? throw new InvalidOperationException("BEAGLE_CSV not set");
        _colony = int.Parse(Environment.GetEnvironmentVariable("BEAGLE_COLONY") ?? "250000");

        var lines = File.ReadAllLines(path);
        var header = lines[0].Split(',');
        _labels = header[..^1];
        var n = lines.Length - 1;
        _inputs = new float[n][];
        _outputs = new float[n];
        for (var i = 0; i < n; i++)
        {
            var cols = lines[i + 1].Split(',');
            _inputs[i] = new float[_labels.Length];
            for (var j = 0; j < _labels.Length; j++) _inputs[i][j] = float.Parse(cols[j]);
            _outputs[i] = float.Parse(cols[^1]);
        }

    }

    // FIX (2026-08-31, "epg0frozen"): the base MLSetup constructor
    // computes MaxGenerationScore = MaxScore * ExperimentsPerGeneration BEFORE
    // this class's constructor body runs. The previous version assigned _epg
    // in the constructor, so the base ctor's virtual call saw 0 ->
    // MaxGenerationScore = 0 -> ASR = Score/0 = Infinity (NaN at Score==0) ->
    // a NaN-ASR organism permanently short-circuits the best-ever archive scan
    // (MLEngine.UpdateMostAccurateOrganismsWith) -> the "Most accurate ever"
    // genome froze at generation 1 in every recorded quality run (empirical:
    // beagle_1e8_run.log, 100 identical lines). The property is now lazily
    // self-computing so its FIRST call — from the base ctor — is correct.
    private uint ComputeEpg()
    {
        var epgEnv = Environment.GetEnvironmentVariable("BEAGLE_EPG");
        var n = _inputs?.Length ?? (File.ReadLines(
            Environment.GetEnvironmentVariable("BEAGLE_CSV")!).Count() - 1);
        var epg = epgEnv != null ? uint.Parse(epgEnv) : (uint)Math.Min(n, 1024);
        epg = Math.Max(32, epg / 32 * 32);   // their tutorial s3.7: divisible by 32
        return Math.Min(epg, 1024);
    }

    public override string Name => "CsvMLSetup-" +
        Path.GetFileNameWithoutExtension(Environment.GetEnvironmentVariable("BEAGLE_CSV") ?? "x");

    public override string[] GetInputLabels() => _labels;

    public override (float[], float) GetNextInputsAndCorrectOutput(float[] inputsToFill)
    {
        var i = (int)((ulong)Interlocked.Increment(ref _cursor) % (ulong)_inputs.Length);
        Array.Copy(_inputs[i], inputsToFill, _labels.Length);
        return (inputsToFill, _outputs[i]);
    }

    public override int TargetColonySize(int generation) => _colony;

    public override uint ExperimentsPerGeneration => _epg != 0 ? _epg : (_epg = ComputeEpg());

    public override long TotalBirthsToResetColonyIfNoProgress => -1;   // no resets

    public override bool KeepOptimizingAfterSolutionFound => true;     // we cut at gen 100

    public override OpEnum[] GetAllowedOperations() =>
        base.GetAllowedOperations().Where(x =>
            x != OpEnum.Sign && x != OpEnum.Sqrt && x != OpEnum.Cbrt &&
            x != OpEnum.Square && x != OpEnum.Cube && x != OpEnum.Ln &&
            x != OpEnum.Tan && x != OpEnum.Arcsin && x != OpEnum.Arccos &&
            x != OpEnum.Arctan && x != OpEnum.Tanh && x != OpEnum.Exp &&
            x != OpEnum.Pow).ToArray();
}
