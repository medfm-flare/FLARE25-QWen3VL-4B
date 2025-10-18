#!/usr/bin/env python3
"""
Real-time training monitoring script for FLARE25 Qwen3-VL training.
Monitors training logs, GPU usage, and provides progress updates.
"""

import os
import sys
import time
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List


class TrainingMonitor:
    """Monitor training progress from log files and system metrics."""

    def __init__(self, output_dir: str, refresh_interval: int = 10):
        self.output_dir = Path(output_dir)
        self.refresh_interval = refresh_interval
        self.start_time = None
        self.last_step = 0
        self.last_loss = None
        self.total_steps = None
        self.losses = []

    def find_log_file(self) -> Optional[Path]:
        """Find the most recent training log file."""
        if not self.output_dir.exists():
            return None

        # Look for trainer_log.jsonl or any .log files
        log_files = list(self.output_dir.glob("*.log"))
        log_files.extend(list(self.output_dir.glob("trainer_log.jsonl")))

        if not log_files:
            return None

        # Return most recent
        return max(log_files, key=lambda p: p.stat().st_mtime)

    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a log line for training metrics."""
        # Look for loss pattern: 'loss': 1.234 or "loss": 1.234
        loss_match = re.search(r"['\"]loss['\"]:\s*([\d.]+)", line)
        # Look for step pattern: 'step': 123 or "step": 123
        step_match = re.search(r"['\"](?:step|global_step)['\"]:\s*(\d+)", line)

        if loss_match or step_match:
            result = {}
            if loss_match:
                result['loss'] = float(loss_match.group(1))
            if step_match:
                result['step'] = int(step_match.group(1))
            return result
        return None

    def get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU utilization stats using nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return None

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'utilization': int(parts[2]),
                            'memory_used': int(parts[3]),
                            'memory_total': int(parts[4]),
                            'temperature': int(parts[5])
                        })

            return {'gpus': gpus} if gpus else None
        except Exception:
            return None

    def format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        return str(timedelta(seconds=int(seconds)))

    def estimate_remaining_time(self, current_step: int, elapsed_seconds: float) -> Optional[str]:
        """Estimate remaining training time."""
        if not self.total_steps or current_step == 0:
            return None

        steps_per_second = current_step / elapsed_seconds
        remaining_steps = self.total_steps - current_step
        remaining_seconds = remaining_steps / steps_per_second

        return self.format_time(remaining_seconds)

    def display_status(self, metrics: Dict, gpu_stats: Optional[Dict]):
        """Display current training status."""
        # Clear screen
        os.system('clear' if os.name != 'nt' else 'cls')

        print("=" * 80)
        print(f"FLARE25 Qwen3-VL Training Monitor")
        print(f"Output Directory: {self.output_dir}")
        print(f"Monitoring since: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Training progress
        print("\n📊 Training Progress:")
        print("-" * 80)

        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"Elapsed Time:     {self.format_time(elapsed)}")

        if metrics.get('step') is not None:
            current_step = metrics['step']
            print(f"Current Step:     {current_step}")

            if self.total_steps:
                progress = (current_step / self.total_steps) * 100
                print(f"Progress:         {progress:.1f}% ({current_step}/{self.total_steps})")

                remaining = self.estimate_remaining_time(current_step, elapsed)
                if remaining:
                    print(f"Est. Remaining:   {remaining}")

        if metrics.get('loss') is not None:
            print(f"Current Loss:     {metrics['loss']:.4f}")

            if len(self.losses) > 1:
                avg_loss = sum(self.losses[-10:]) / len(self.losses[-10:])
                print(f"Avg Loss (10):    {avg_loss:.4f}")

        # GPU stats
        if gpu_stats and 'gpus' in gpu_stats:
            print("\n🖥️  GPU Status:")
            print("-" * 80)
            for gpu in gpu_stats['gpus']:
                print(f"GPU {gpu['index']}: {gpu['name']}")
                print(f"  Utilization:  {gpu['utilization']}%")
                print(f"  Memory:       {gpu['memory_used']} MB / {gpu['memory_total']} MB "
                      f"({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
                print(f"  Temperature:  {gpu['temperature']}°C")

        # Recent losses
        if self.losses:
            print("\n📉 Recent Losses (last 10):")
            print("-" * 80)
            recent = self.losses[-10:]
            loss_str = ", ".join([f"{l:.4f}" for l in recent])
            print(loss_str)

        print("\n" + "=" * 80)
        print(f"Refreshing every {self.refresh_interval} seconds... (Press Ctrl+C to exit)")
        print("=" * 80)

    def monitor(self):
        """Main monitoring loop."""
        print("Starting training monitor...")
        print(f"Watching: {self.output_dir}")
        print(f"Refresh interval: {self.refresh_interval}s")
        print("\nWaiting for training to start...\n")

        self.start_time = time.time()
        last_file_size = 0

        try:
            while True:
                log_file = self.find_log_file()

                if log_file and log_file.exists():
                    # Check if file has new content
                    current_size = log_file.stat().st_size

                    if current_size > last_file_size:
                        # Read new lines
                        with open(log_file, 'r') as f:
                            f.seek(last_file_size)
                            new_lines = f.readlines()

                        # Parse new lines for metrics
                        for line in new_lines:
                            parsed = self.parse_log_line(line)
                            if parsed:
                                if 'step' in parsed:
                                    self.last_step = parsed['step']
                                if 'loss' in parsed:
                                    self.last_loss = parsed['loss']
                                    self.losses.append(self.last_loss)

                        last_file_size = current_size

                # Get GPU stats
                gpu_stats = self.get_gpu_stats()

                # Display status
                metrics = {
                    'step': self.last_step if self.last_step > 0 else None,
                    'loss': self.last_loss
                }
                self.display_status(metrics, gpu_stats)

                # Wait before next refresh
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user.")
            print("=" * 80)
            print("\nFinal Status:")
            if self.last_step > 0:
                print(f"  Last Step: {self.last_step}")
            if self.last_loss is not None:
                print(f"  Last Loss: {self.last_loss:.4f}")
            if self.losses:
                avg_loss = sum(self.losses) / len(self.losses)
                print(f"  Average Loss: {avg_loss:.4f}")
            print("=" * 80)

    def tail_logs(self, num_lines: int = 50):
        """Display last N lines of training logs."""
        log_file = self.find_log_file()

        if not log_file or not log_file.exists():
            print(f"❌ No log file found in {self.output_dir}")
            return

        print(f"Last {num_lines} lines from {log_file.name}:")
        print("=" * 80)

        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-num_lines:]:
                print(line.rstrip())


def main():
    parser = argparse.ArgumentParser(
        description="Monitor FLARE25 Qwen3-VL training progress"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/qwen3vl_flare25_starter',
        help='Training output directory to monitor'
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=10,
        help='Refresh interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--tail',
        type=int,
        default=None,
        help='Just print last N lines of log and exit'
    )

    args = parser.parse_args()

    monitor = TrainingMonitor(
        output_dir=args.output_dir,
        refresh_interval=args.refresh
    )

    if args.tail:
        monitor.tail_logs(num_lines=args.tail)
    else:
        monitor.monitor()


if __name__ == "__main__":
    main()
