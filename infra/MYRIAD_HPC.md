# Running SPACE Pipeline on UCL Myriad HPC

## Why Myriad?

The SPACE pipeline requires significant CPU resources for several days to process the full plasmid dataset (~375,000 files, ~56GB). Key resource-intensive steps:

- **Bakta**: Bacterial genome annotation - slow and memory-intensive
- **MOB-suite**: Plasmid typing and mobility prediction
- **COPLA**: Plasmid classification

Myriad provides:
- Up to 36 CPUs per node
- Up to 1.5TB RAM on high-memory nodes
- SGE job scheduler for parallel processing
- Nextflow can submit 100+ jobs simultaneously

## Connection Setup

### SSH Configuration

Add to `~/.ssh/config`:

```
Host ucl-gateway
    HostName ssh-gateway.ucl.ac.uk
    User YOUR_UCL_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 1h

Host myriad
    HostName myriad.rc.ucl.ac.uk
    User YOUR_UCL_USERNAME
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump ucl-gateway
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 1h
    Compression yes
```

Create the sockets directory:
```bash
mkdir -p ~/.ssh/sockets
```

### VPN Requirement

**Important**: UCL's SSH gateway requires VPN access from outside the UCL network. The jump node configuration works, but you must be connected to the UCL VPN first.

### Connecting

```bash
# Add SSH key to agent
ssh-add ~/.ssh/id_ed25519

# Connect to Myriad
ssh myriad
```

## Directory Structure on Myriad

All work happens in Scratch (temporary storage, auto-deleted after 30 days of inactivity):

```
~/Scratch/
├── nextflow                    # Nextflow binary
├── PlasmidLLM/                 # Pipeline code
│   └── src/nextflow/           # Nextflow pipeline
│       ├── main.nf
│       └── nextflow.config
├── databases/
│   └── bakta/                  # Bakta database (~3.7GB)
├── .apptainer/
│   ├── cache/                  # Container images (.sif files)
│   │   ├── bakta_latest.sif
│   │   ├── mob_suite_3.0.3.sif
│   │   └── copla_1.0.sif
│   └── tmp/                    # Build temp (can get large)
└── results/                    # Pipeline output
```

## Initial Setup (One-Time)

### 1. Install Nextflow

```bash
cd ~/Scratch
curl -s https://get.nextflow.io | bash
chmod +x nextflow
```

### 2. Create Conda Environment

```bash
source /etc/profile.d/modules.sh
source /shared/ucl/apps/miniconda/24.3.0-0/etc/profile.d/conda.sh

conda create -n space python=3.11 biopython pandas pyarrow -y
conda activate space
pip install awscli
```

### 3. Transfer Pipeline Code

From your local machine:
```bash
rsync -avz /path/to/PlasmidLLM/ myriad:~/Scratch/PlasmidLLM/
```

### 4. Transfer Bakta Database

The Bakta database is large (~3.7GB). Transfer from local or download:

```bash
# Option A: Transfer from local
rsync -avz ~/Downloads/db-light/ myriad:~/Scratch/databases/bakta/

# Option B: Download on Myriad (submit as job)
# See container pull job script below
```

### 5. Pull Container Images

Containers must be pulled on compute nodes (not login nodes). Submit a job:

```bash
cat > ~/Scratch/pull_containers.sh << 'EOF'
#!/bin/bash
#$ -l h_rt=4:0:0
#$ -l h_vmem=16G,mem_free=16G
#$ -pe smp 1
#$ -N pull_containers

source /etc/profile.d/modules.sh
module load apptainer

export APPTAINER_TMPDIR=$HOME/Scratch/.apptainer/tmp
export APPTAINER_CACHEDIR=$HOME/Scratch/.apptainer/cache

mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

echo "Pulling Bakta..."
apptainer pull --force $APPTAINER_CACHEDIR/bakta_latest.sif docker://oschwengers/bakta:latest

echo "Pulling MOB-suite..."
apptainer pull --force $APPTAINER_CACHEDIR/mob_suite_3.0.3.sif docker://kbessonov/mob_suite:3.0.3

echo "Pulling COPLA..."
apptainer pull --force $APPTAINER_CACHEDIR/copla_1.0.sif docker://rpalcab/copla:1.0

echo "Done!"
ls -lh $APPTAINER_CACHEDIR/*.sif
EOF

qsub ~/Scratch/pull_containers.sh
```

Monitor with: `qstat -u $USER`

### 6. Set Up AWS Credentials

Copy your AWS credentials to Myriad:
```bash
rsync -avz ~/.aws/ myriad:~/.aws/
```

## Running the Pipeline

### With Local Test Data

```bash
ssh myriad
cd ~/Scratch/PlasmidLLM/src/nextflow

source /etc/profile.d/modules.sh
source /shared/ucl/apps/miniconda/24.3.0-0/etc/profile.d/conda.sh
conda activate space

~/Scratch/nextflow run main.nf -profile myriad
```

### With S3 Data (Full Dataset)

```bash
~/Scratch/nextflow run main.nf -profile myriad_s3 -resume
```

### Key Flags

- `-resume`: Resume from previous run (critical for long-running jobs)
- `-with-tower`: Monitor on Seqera Platform (token in config)
- `--skip_bakta`: Skip Bakta annotation
- `--skip_mobsuite`: Skip MOB-suite typing
- `--skip_copla`: Skip COPLA classification

## S3 Data Location

Data is stored in AWS S3:

```
s3://phd-research-storage-1758274488/plasmid_data_20251209/
├── 0_raw/
│   ├── GenBank/GenBank/gbk/     # 17,700 files
│   ├── RefSeq/
│   ├── PLSDB/
│   ├── addgene/
│   └── mMGE/
└── pipeline_output/              # Results written here
```

Total: ~375,000 files, ~56GB

## Monitoring Jobs

```bash
# View your jobs
qstat -u $USER

# View job details
qstat -j JOB_ID

# View job output
cat ~/JOB_NAME.oJOB_ID

# View job errors
cat ~/JOB_NAME.eJOB_ID

# Cancel a job
qdel JOB_ID
```

## Seqera Platform Monitoring

The pipeline is configured to report to Seqera Platform (formerly Nextflow Tower). Monitor runs at:

https://cloud.seqera.io/user/mcclain-thiel/watch/

## Troubleshooting

### Connection Issues

1. Ensure VPN is connected
2. Check SSH key is loaded: `ssh-add -l`
3. Test gateway first: `ssh ucl-gateway "hostname"`

### Container Build Failures

If containers fail with "Out of memory":
- Request more memory in job script (`-l h_vmem=16G,mem_free=16G`)
- Clean old temp files: `rm -rf ~/Scratch/.apptainer/tmp/build-temp-*`

### Pipeline Failures

1. Check Nextflow log: `.nextflow.log`
2. Check work directory for task logs: `work/XX/HASH/.command.log`
3. Resume failed runs: `nextflow run main.nf -resume`

### Disk Space

Scratch has limited space. Clean up periodically:
```bash
# Check usage
du -sh ~/Scratch/*

# Clean Nextflow work directory (after successful run)
rm -rf ~/Scratch/PlasmidLLM/src/nextflow/work

# Clean container build temps
rm -rf ~/Scratch/.apptainer/tmp/build-temp-*
```

## Useful Links

- [UCL Myriad Documentation](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/)
- [UCL Remote Access](https://www.rc.ucl.ac.uk/docs/Remote_Access/)
- [nf-core UCL Myriad Config](https://nf-co.re/configs/ucl_myriad/)
- [Nextflow Documentation](https://www.nextflow.io/docs/latest/)
