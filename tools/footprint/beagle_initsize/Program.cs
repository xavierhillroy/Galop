using Run.MLSetups;
// Initial-organism size probe: build organisms exactly as MLEngine's "Create Initial Colony"
// does (Organism.CreateByRandomLoadOrConstCommandThenMutate with the setup's allowed operations)
// and report the mean/min/max command count. Stock BeagleLib, unmodified.
using BeagleLib.Agent;
using BeagleLib.Engine;
using BeagleLib.VM;
var setup = new CsvMLSetup();                       // registers itself as MLSetup.Current (adapter precedent)
var ops = MLSetup.Current.GetAllowedOperations();
var adj = MLSetup.Current.GetAllowedAdjunctOperationsCount();
int n = args.Length > 0 ? int.Parse(args[0]) : 20000;
byte inputs = (byte)(args.Length > 1 ? int.Parse(args[1]) : 1);
long sum = 0; int min = int.MaxValue, max = 0; var hist = new int[64];
for (int i = 0; i < n; i++) {
    var o = Organism.CreateByRandomLoadOrConstCommandThenMutate(inputs, ops, adj);
    int L = o.Commands.Length; sum += L; if (L < min) min = L; if (L > max) max = L; if (L < 64) hist[L]++;
}
Console.WriteLine($"initial organisms: n={n} inputs={inputs} allowedOps={ops.Length} mean {sum/(double)n:F2} commands (min {min}, max {max}); sizeof(Command)={System.Runtime.CompilerServices.Unsafe.SizeOf<Command>()} B");
Console.WriteLine("length histogram (1..12): " + string.Join(" ", Enumerable.Range(1, 12).Select(L => $"{L}:{hist[L]}")));
