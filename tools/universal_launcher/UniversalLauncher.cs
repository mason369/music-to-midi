using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Management;
using System.Text;
using System.Windows.Forms;

namespace MusicToMidi.UniversalLauncher
{
    internal static class Program
    {
        private const string AcceleratorVariable = "MUSIC_TO_MIDI_ACCELERATOR";
        private const string TraceFileVariable = "MUSIC_TO_MIDI_UNIVERSAL_TRACE_FILE";
        private const string DisableDialogVariable = "MUSIC_TO_MIDI_UNIVERSAL_NO_DIALOG";

        [STAThread]
        private static int Main(string[] args)
        {
            bool backend = IsBackendLauncher();
            try
            {
                string accelerator = ResolveAccelerator();
                string launcherDirectory = AppDomain.CurrentDomain.BaseDirectory;
                string runtimeDirectory = Path.Combine(
                    launcherDirectory,
                    "runtimes",
                    accelerator
                );
                string childName = GetChildExecutableName(backend, accelerator);
                string childPath = Path.Combine(runtimeDirectory, childName);
                RequireExecutable(childPath);
                WriteSelectionTrace(backend, accelerator, childPath);

                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = childPath,
                    Arguments = JoinArguments(args),
                    WorkingDirectory = runtimeDirectory,
                    UseShellExecute = false,
                };
                startInfo.EnvironmentVariables[AcceleratorVariable] = accelerator;

                using (Process child = Process.Start(startInfo))
                {
                    if (child == null)
                    {
                        throw new InvalidOperationException(
                            "Windows 没有返回已启动的子进程。"
                        );
                    }
                    child.WaitForExit();
                    return child.ExitCode;
                }
            }
            catch (Exception ex)
            {
                ReportFailure(backend, ex.Message);
                return 70;
            }
        }

        private static bool IsBackendLauncher()
        {
            string launcherName = Path.GetFileNameWithoutExtension(
                Process.GetCurrentProcess().MainModule.FileName
            );
            return launcherName.IndexOf(
                "Backend",
                StringComparison.OrdinalIgnoreCase
            ) >= 0;
        }

        private static string ResolveAccelerator()
        {
            string explicitAccelerator = (
                Environment.GetEnvironmentVariable(AcceleratorVariable) ?? string.Empty
            ).Trim().ToLowerInvariant();
            if (explicitAccelerator.Length > 0)
            {
                if (explicitAccelerator != "cuda" && explicitAccelerator != "xpu")
                {
                    throw new InvalidOperationException(
                        AcceleratorVariable
                        + " 只能是 cuda 或 xpu，实际值为："
                        + explicitAccelerator
                    );
                }
                return explicitAccelerator;
            }

            List<string> deviceNames = ReadVideoControllerNames();
            bool hasNvidia = false;
            bool hasSupportedIntelCandidate = false;
            foreach (string name in deviceNames)
            {
                if (name.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    hasNvidia = true;
                }
                if (
                    name.IndexOf("Intel", StringComparison.OrdinalIgnoreCase) >= 0
                    && name.IndexOf("Arc", StringComparison.OrdinalIgnoreCase) >= 0
                )
                {
                    hasSupportedIntelCandidate = true;
                }
            }

            // A host containing both vendors selects the validated CUDA baseline.
            // This is a deterministic preference, not a retry/failure fallback.
            if (hasNvidia)
            {
                return "cuda";
            }
            if (hasSupportedIntelCandidate)
            {
                return "xpu";
            }

            throw new InvalidOperationException(
                "未检测到可选择的 NVIDIA CUDA 或 Intel Arc XPU 设备。"
                + "检测结果："
                + string.Join(" | ", deviceNames.ToArray())
            );
        }

        private static List<string> ReadVideoControllerNames()
        {
            List<string> names = new List<string>();
            using (ManagementObjectSearcher searcher = new ManagementObjectSearcher(
                "SELECT Name FROM Win32_VideoController"
            ))
            using (ManagementObjectCollection devices = searcher.Get())
            {
                foreach (ManagementObject device in devices)
                {
                    object rawName = device["Name"];
                    if (rawName == null)
                    {
                        continue;
                    }
                    string name = rawName.ToString().Trim();
                    if (name.Length > 0)
                    {
                        names.Add(name);
                    }
                }
            }
            if (names.Count == 0)
            {
                throw new InvalidOperationException(
                    "Win32_VideoController 没有返回任何显卡。"
                );
            }
            return names;
        }

        private static string GetChildExecutableName(bool backend, string accelerator)
        {
            if (backend)
            {
                return accelerator == "xpu"
                    ? "MusicToMidiBackendXpu.exe"
                    : "MusicToMidiBackend.exe";
            }
            return accelerator == "xpu" ? "MusicToMidiXpu.exe" : "MusicToMidi.exe";
        }

        private static void RequireExecutable(string path)
        {
            FileInfo file = new FileInfo(path);
            if (!file.Exists || file.Length <= 0)
            {
                throw new FileNotFoundException(
                    "所选加速器的便携运行入口不存在或为空：" + path,
                    path
                );
            }
        }

        private static void WriteSelectionTrace(
            bool backend,
            string accelerator,
            string childPath
        )
        {
            string tracePath = (
                Environment.GetEnvironmentVariable(TraceFileVariable) ?? string.Empty
            ).Trim();
            if (tracePath.Length == 0)
            {
                return;
            }
            string fullPath = Path.GetFullPath(tracePath);
            string parent = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(parent))
            {
                Directory.CreateDirectory(parent);
            }
            string line = string.Join(
                "\t",
                new string[]
                {
                    DateTimeOffset.Now.ToString("o"),
                    backend ? "backend" : "app",
                    accelerator,
                    childPath,
                }
            );
            File.AppendAllText(fullPath, line + Environment.NewLine, new UTF8Encoding(false));
        }

        private static string JoinArguments(string[] args)
        {
            List<string> quoted = new List<string>();
            foreach (string arg in args)
            {
                quoted.Add(QuoteWindowsArgument(arg));
            }
            return string.Join(" ", quoted.ToArray());
        }

        private static string QuoteWindowsArgument(string value)
        {
            if (value.Length == 0)
            {
                return "\"\"";
            }
            if (
                value.IndexOfAny(new char[] { ' ', '\t', '\n', '\v', '\"' }) < 0
            )
            {
                return value;
            }

            StringBuilder result = new StringBuilder();
            result.Append('\"');
            int backslashes = 0;
            foreach (char character in value)
            {
                if (character == '\\')
                {
                    backslashes++;
                    continue;
                }
                if (character == '\"')
                {
                    result.Append('\\', backslashes * 2 + 1);
                    result.Append('\"');
                    backslashes = 0;
                    continue;
                }
                result.Append('\\', backslashes);
                backslashes = 0;
                result.Append(character);
            }
            result.Append('\\', backslashes * 2);
            result.Append('\"');
            return result.ToString();
        }

        private static void ReportFailure(bool backend, string message)
        {
            string fullMessage = "MusicToMidi Universal 启动失败：" + message;
            Console.Error.WriteLine(fullMessage);
#if APP_LAUNCHER
            if (
                !string.Equals(
                    Environment.GetEnvironmentVariable(DisableDialogVariable),
                    "1",
                    StringComparison.Ordinal
                )
            )
            {
                MessageBox.Show(
                    fullMessage,
                    "MusicToMidi Universal",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
#endif
        }
    }
}
