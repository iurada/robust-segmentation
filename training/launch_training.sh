#!/usr/bin/env bash
H=$(hostname)

JOB_NAME="${JOB_NAME:-inet}"
NGPUS=${NGPUS:-1}
MAX_HOURS=${MAX_HOURS:-24}

function die()
{
	echo $1
	exit 1
}

case $H in
  *"franklin"*) # Franklin
    echo "Launching on Franklin"

    if [ "$ARRAY" != "" ] ; then
      die "Arrays not supported on this system"
    fi

    ARGS="--num_gpus $NGPUS $@"
    qsub \
      -N "$JOB_NAME" \
      -v "ARGS=$ARGS" \
      -l "walltime=${MAX_HOURS}:00:00" \
      -l "select=1:ncpus=$((NGPUS * 4)):ngpus=${NGPUS}:scratch=170gb" \
      job_files/franklin_job.job
  ;;
  *"login"*) # Marconi100
    echo "Launching on Marconi100"

    if [ "$ARRAY" != "" ] ; then
      echo "Launching ${ARRAY} sequential jobs"
      ARRAY="--array=1-$ARRAY%1"
    fi

    if [ "$QOS" != "" ] ; then
      echo "Setting QOS to: ${QOS}"
      QOS="--qos=${QOS}"
    fi

    sbatch \
      -A "${ACCOUNT:-IscrC_ReAcT}" \
      --partition "${PARTITION:-m100_usr_prod}" \
      -N 1 \
      --job-name="$JOB_NAME" \
      --gres="gpu:$NGPUS" \
      --time="${MAX_HOURS}:00:00" \
      --cpus-per-task=$((NGPUS * 8)) \
      --ntasks-per-node=1 \
      --mem="$((NGPUS * 61500))M" \
      $ARRAY $QOS \
      job_files/marconi100_job.job --num_gpus "$NGPUS" $@
  ;;
  *"dgxslurm"*)
    echo "Launching on DGX @ Cineca"

    if [ "$ARRAY" != "" ] ; then
      echo "Launching ${ARRAY} sequential jobs"
      ARRAY="--array=1-$ARRAY%1"
    fi

    if [ "$QOS" != "" ] ; then
      echo "Setting QOS to: ${QOS}"
      QOS="--qos=${QOS}"
    fi

    sbatch \
      -A "${ACCOUNT:-IscrC_Tiny-NAS_0}" \
      --partition "${PARTITION:-dgx_usr_preempt}" \
      -N 1 \
      --job-name="$JOB_NAME" \
      --gres="gpu:$NGPUS" \
      --time="${MAX_HOURS}:00:00" \
      --cpus-per-task=$((NGPUS * 16)) \
      --ntasks-per-node=1 \
      --mem="$((NGPUS * 100000))M" \
      $ARRAY $QOS \
      job_files/dgx_job.job --num_gpus "$NGPUS" $@
  ;;
  *"legion"*) # Legion
    echo "Launching on Legion"
    die "Not implemented"
  ;;
  *"frontend"*) # Cluster nostro
    echo "Launching on our cluster"
    die "Not implemented"
  ;;
  *) # Altro (macchine lab, ecc)
    echo "Launching on ${H^} (generic)"
    ./multi_run.sh --num_gpus "$NGPUS" $@
  ;;
esac
