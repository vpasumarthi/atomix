"""Job submission abstraction for atomix."""

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class JobSubmitter(ABC):
    """Abstract base class for job submission systems.

    Parameters
    ----------
    working_dir : Path | str
        Directory where calculation files are located.
    """

    def __init__(self, working_dir: Path | str = ".") -> None:
        self.working_dir = Path(working_dir)

    @abstractmethod
    def submit(self, **kwargs: Any) -> str:
        """Submit job and return job ID.

        Returns
        -------
        str
            Job ID or process identifier.
        """
        pass

    @abstractmethod
    def status(self, job_id: str) -> str:
        """Check job status.

        Parameters
        ----------
        job_id : str
            Job identifier.

        Returns
        -------
        str
            Status: 'pending', 'running', 'completed', 'failed', 'unknown'.
        """
        pass

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Parameters
        ----------
        job_id : str
            Job identifier.

        Returns
        -------
        bool
            True if cancellation was successful.
        """
        pass

    def write_script(self, script_path: Path, content: str) -> Path:
        """Write job script to file.

        Parameters
        ----------
        script_path : Path
            Path to write script.
        content : str
            Script content.

        Returns
        -------
        Path
            Path to written script.
        """
        script_path.write_text(content)
        script_path.chmod(0o755)
        return script_path


class SLURMSubmitter(JobSubmitter):
    """SLURM job submission.

    Parameters
    ----------
    working_dir : Path | str
        Directory where calculation files are located.
    partition : str | None
        SLURM partition name.
    account : str | None
        SLURM account name.
    nodes : int
        Number of nodes.
    ntasks_per_node : int
        Tasks per node.
    time : str
        Wall time (e.g., "24:00:00").
    """

    def __init__(
        self,
        working_dir: Path | str = ".",
        partition: str | None = None,
        account: str | None = None,
        nodes: int = 1,
        ntasks_per_node: int = 32,
        time: str = "24:00:00",
        **kwargs: Any,
    ) -> None:
        super().__init__(working_dir)
        self.partition = partition
        self.account = account
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.time = time
        self.extra_options = kwargs

    def generate_script(
        self,
        job_name: str = "vasp_job",
        vasp_command: str = "vasp_std",
        modules: list[str] | None = None,
        pre_commands: list[str] | None = None,
    ) -> str:
        """Generate SLURM submission script.

        Parameters
        ----------
        job_name : str
            Name for the job.
        vasp_command : str
            VASP executable command.
        modules : list[str] | None
            Modules to load.
        pre_commands : list[str] | None
            Commands to run before VASP.

        Returns
        -------
        str
            SLURM script content.
        """
        lines = ["#!/bin/bash"]

        # SLURM directives
        lines.append(f"#SBATCH --job-name={job_name}")
        lines.append(f"#SBATCH --nodes={self.nodes}")
        lines.append(f"#SBATCH --ntasks-per-node={self.ntasks_per_node}")
        lines.append(f"#SBATCH --time={self.time}")
        lines.append(f"#SBATCH --output={job_name}_%j.out")
        lines.append(f"#SBATCH --error={job_name}_%j.err")

        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")

        # Extra SBATCH options
        for key, value in self.extra_options.items():
            lines.append(f"#SBATCH --{key}={value}")

        lines.append("")

        # Module loads
        if modules:
            for module in modules:
                lines.append(f"module load {module}")
            lines.append("")

        # Pre-commands
        if pre_commands:
            lines.extend(pre_commands)
            lines.append("")

        # Change to working directory
        lines.append(f"cd {self.working_dir.absolute()}")
        lines.append("")

        # Run VASP
        ntasks = self.nodes * self.ntasks_per_node
        lines.append(f"srun -n {ntasks} {vasp_command}")

        return "\n".join(lines) + "\n"

    def submit(
        self,
        job_name: str = "vasp_job",
        vasp_command: str = "vasp_std",
        modules: list[str] | None = None,
        pre_commands: list[str] | None = None,
        script_name: str = "submit.slurm",
    ) -> str:
        """Submit SLURM job.

        Returns
        -------
        str
            SLURM job ID.
        """
        # Generate and write script
        script_content = self.generate_script(
            job_name=job_name,
            vasp_command=vasp_command,
            modules=modules,
            pre_commands=pre_commands,
        )
        script_path = self.working_dir / script_name
        self.write_script(script_path, script_content)

        # Submit job
        result = subprocess.run(
            ["sbatch", str(script_path)],
            cwd=self.working_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")

        # Parse job ID from output: "Submitted batch job 12345"
        output = result.stdout.strip()
        job_id = output.split()[-1]
        return job_id

    def status(self, job_id: str) -> str:
        """Check SLURM job status."""
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Job not in queue - check if completed
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-o", "State", "-X"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                state = result.stdout.strip().upper()
                if "COMPLETED" in state:
                    return "completed"
                elif "FAILED" in state or "CANCELLED" in state:
                    return "failed"
            return "unknown"

        state = result.stdout.strip().upper()
        if state in ("PENDING", "CONFIGURING"):
            return "pending"
        elif state in ("RUNNING", "COMPLETING"):
            return "running"
        elif state == "COMPLETED":
            return "completed"
        elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            return "failed"
        return "unknown"

    def cancel(self, job_id: str) -> bool:
        """Cancel SLURM job."""
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0


class PBSSubmitter(JobSubmitter):
    """PBS/Torque job submission.

    Parameters
    ----------
    working_dir : Path | str
        Directory where calculation files are located.
    queue : str | None
        PBS queue name.
    nodes : int
        Number of nodes.
    ppn : int
        Processors per node.
    walltime : str
        Wall time (e.g., "24:00:00").
    """

    def __init__(
        self,
        working_dir: Path | str = ".",
        queue: str | None = None,
        nodes: int = 1,
        ppn: int = 32,
        walltime: str = "24:00:00",
        **kwargs: Any,
    ) -> None:
        super().__init__(working_dir)
        self.queue = queue
        self.nodes = nodes
        self.ppn = ppn
        self.walltime = walltime
        self.extra_options = kwargs

    def generate_script(
        self,
        job_name: str = "vasp_job",
        vasp_command: str = "vasp_std",
        modules: list[str] | None = None,
        pre_commands: list[str] | None = None,
    ) -> str:
        """Generate PBS submission script."""
        lines = ["#!/bin/bash"]

        # PBS directives
        lines.append(f"#PBS -N {job_name}")
        lines.append(f"#PBS -l nodes={self.nodes}:ppn={self.ppn}")
        lines.append(f"#PBS -l walltime={self.walltime}")
        lines.append(f"#PBS -o {job_name}.out")
        lines.append(f"#PBS -e {job_name}.err")

        if self.queue:
            lines.append(f"#PBS -q {self.queue}")

        # Extra PBS options
        for key, value in self.extra_options.items():
            lines.append(f"#PBS -{key} {value}")

        lines.append("")

        # Module loads
        if modules:
            for module in modules:
                lines.append(f"module load {module}")
            lines.append("")

        # Pre-commands
        if pre_commands:
            lines.extend(pre_commands)
            lines.append("")

        # Change to working directory
        lines.append(f"cd {self.working_dir.absolute()}")
        lines.append("")

        # Run VASP
        ntasks = self.nodes * self.ppn
        lines.append(f"mpirun -np {ntasks} {vasp_command}")

        return "\n".join(lines) + "\n"

    def submit(
        self,
        job_name: str = "vasp_job",
        vasp_command: str = "vasp_std",
        modules: list[str] | None = None,
        pre_commands: list[str] | None = None,
        script_name: str = "submit.pbs",
    ) -> str:
        """Submit PBS job."""
        script_content = self.generate_script(
            job_name=job_name,
            vasp_command=vasp_command,
            modules=modules,
            pre_commands=pre_commands,
        )
        script_path = self.working_dir / script_name
        self.write_script(script_path, script_content)

        result = subprocess.run(
            ["qsub", str(script_path)],
            cwd=self.working_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"PBS submission failed: {result.stderr}")

        # PBS returns job ID like "12345.hostname"
        job_id = result.stdout.strip()
        return job_id

    def status(self, job_id: str) -> str:
        """Check PBS job status."""
        result = subprocess.run(
            ["qstat", "-f", job_id],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return "unknown"

        output = result.stdout
        if "job_state = Q" in output:
            return "pending"
        elif "job_state = R" in output:
            return "running"
        elif "job_state = C" in output:
            return "completed"
        elif "job_state = E" in output:
            return "failed"
        return "unknown"

    def cancel(self, job_id: str) -> bool:
        """Cancel PBS job."""
        result = subprocess.run(
            ["qdel", job_id],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0


class LocalRunner(JobSubmitter):
    """Local job execution (for testing or small jobs).

    Parameters
    ----------
    working_dir : Path | str
        Directory where calculation files are located.
    """

    def __init__(self, working_dir: Path | str = ".") -> None:
        super().__init__(working_dir)
        self._processes: dict[str, subprocess.Popen] = {}
        self._counter = 0

    def submit(
        self,
        vasp_command: str = "vasp_std",
        nprocs: int = 1,
        background: bool = True,
        **kwargs: Any,
    ) -> str:
        """Run VASP locally.

        Parameters
        ----------
        vasp_command : str
            VASP executable.
        nprocs : int
            Number of MPI processes.
        background : bool
            Run in background (non-blocking).

        Returns
        -------
        str
            Process ID as string.
        """
        if nprocs > 1:
            cmd = ["mpirun", "-np", str(nprocs), vasp_command]
        else:
            cmd = [vasp_command]

        stdout_file = self.working_dir / "vasp.out"
        stderr_file = self.working_dir / "vasp.err"

        with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
            if background:
                proc = subprocess.Popen(
                    cmd,
                    cwd=self.working_dir,
                    stdout=out,
                    stderr=err,
                )
                self._counter += 1
                job_id = f"local_{self._counter}_{proc.pid}"
                self._processes[job_id] = proc
                return job_id
            else:
                result = subprocess.run(
                    cmd,
                    cwd=self.working_dir,
                    stdout=out,
                    stderr=err,
                )
                self._counter += 1
                job_id = f"local_{self._counter}_done"
                return job_id

    def status(self, job_id: str) -> str:
        """Check local job status."""
        if job_id.endswith("_done"):
            # Check if OUTCAR exists and has "reached required accuracy"
            outcar = self.working_dir / "OUTCAR"
            if outcar.exists():
                content = outcar.read_text()
                if "reached required accuracy" in content:
                    return "completed"
                elif "VERY BAD NEWS" in content or "Error" in content:
                    return "failed"
            return "completed"  # Assume completed if no OUTCAR check

        if job_id not in self._processes:
            return "unknown"

        proc = self._processes[job_id]
        poll = proc.poll()

        if poll is None:
            return "running"
        elif poll == 0:
            return "completed"
        else:
            return "failed"

    def cancel(self, job_id: str) -> bool:
        """Cancel local job."""
        if job_id not in self._processes:
            return False

        proc = self._processes[job_id]
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
            return True
        return False


def get_submitter(
    scheduler: str,
    working_dir: Path | str = ".",
    **kwargs: Any,
) -> JobSubmitter:
    """Factory function to get appropriate job submitter.

    Parameters
    ----------
    scheduler : str
        Scheduler type: 'slurm', 'pbs', 'local'.
    working_dir : Path | str
        Working directory.
    **kwargs
        Scheduler-specific options.

    Returns
    -------
    JobSubmitter
        Appropriate submitter instance.
    """
    scheduler = scheduler.lower()
    if scheduler == "slurm":
        return SLURMSubmitter(working_dir, **kwargs)
    elif scheduler in ("pbs", "torque"):
        return PBSSubmitter(working_dir, **kwargs)
    elif scheduler == "local":
        return LocalRunner(working_dir)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")
