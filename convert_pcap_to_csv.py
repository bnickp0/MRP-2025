
import os
import subprocess

# Path to your folder containing .pcap files
pcap_folder = r"C:\path\to\pcap_files"
output_folder = r"C:\path\to\csv_output"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(pcap_folder):
    if filename.endswith(".pcap"):
        input_path = os.path.join(pcap_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".pcap", ".csv"))
        command = [
            "tshark", "-r", input_path,
            "-T", "fields",
            "-e", "frame.time", "-e", "ip.src", "-e", "ip.dst", "-e", "ip.proto",
            "-e", "tcp.srcport", "-e", "tcp.dstport",
            "-e", "udp.srcport", "-e", "udp.dstport", "-e", "frame.len",
            "-E", "separator=,", "-E", "quote=d", "-E", "header=y"
        ]
        with open(output_path, "w") as f:
            subprocess.run(command, stdout=f)
