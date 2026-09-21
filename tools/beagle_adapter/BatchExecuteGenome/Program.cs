// Batch genome executor for the gcl_eval validation — OUR shim, THEIR engine.
// Links stock BeagleLib unmodified (the CsvMLSetup precedent): parsing is
// their Organism.CreateFromAnyString, evaluation is their CodeMachine (the
// same class their engine evaluates organisms with). Exists because their
// interactive Utils/ExecuteGenome pretty-prints the expression before the
// input loop, and MathExpr.AsTraditionalString() recurses to a StackOverflow
// on deep evolved genomes (Dup/Copy expand the RPN into an exponential tree)
// — killing the process before a single row is evaluated.
// Protocol (same as their utility, minus banners and prompts):
//   stdin:  genome line; one ignored label line per loaded input; then one
//           float per line per input, repeated per row
//   stdout: "Output: <float>" per row.  EOF on stdin: clean exit.
using BeagleLib.Agent;
using BeagleLib.VM;

var gcl = Console.ReadLine();
if (gcl == null) return;
var organism = Organism.CreateFromAnyString(gcl);
var n = organism.Commands.GetMaxInputIdx() + 1;
Console.WriteLine($"Inputs: {n}");                // driver feeds THIS many
for (var i = 0; i < n; i++) Console.ReadLine();   // label lines, ignored
var inputs = new float[n];
while (true)
{
    for (var i = 0; i < n; i++)
    {
        var line = Console.ReadLine();
        if (line == null) return;                 // EOF: done
        inputs[i] = float.Parse(line);
    }
    var output = new CodeMachine().RunCommands(inputs, organism.Commands);
    Console.WriteLine($"Output: {output}");
}
