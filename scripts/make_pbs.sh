#!/bin/bash

function create_pbs_script() {
    local username log_dir="log" ncpus="4" mem="64gb" walltime="24:00:00" queue="normal" project_id="11003281"

    while getopts "u:l:n:m:w:q:p:" opt; do
        case ${opt} in
            u)
                username=$OPTARG
                ;;
            l)
                log_dir=$OPTARG
                ;;
            n)
                ncpus=$OPTARG
                ;;
            m)
                mem=$OPTARG
                ;;
            w)
                walltime=$OPTARG
                ;;
            q)
                queue=$OPTARG
                ;;
            p)
                project_id=$OPTARG
                ;;
            *)
                echo "Invalid option: $OPTARG" 1>&2
                exit 1
                ;;
        esac
    done
    shift $((OPTIND -1))

    if [ -z "$username" ]; then
        echo "Error: Username is a required argument."
        echo "Usage: create_pbs_script -u <username> [-l <log_dir>] [-n <ncpus>] [-m <mem>] [-w <walltime>] [-q <queue>] [-p <project_id>]"
        exit 1
    fi

    cat <<EOF > run_mlflow.pbs
    #!/bin/bash
    #PBS -N run_mlflow
    #PBS -l select=1:ncpus=$ncpus:mem=$mem
    #PBS -l walltime=$walltime
    #PBS -q $queue
    #PBS -koed
    #PBS -o /home/users/$username/$log_dir
    #PBS -e /home/users/$username/$log_dir
    #PBS -P $project_id

    print_hostname() {
        echo "Print Hostname"
        hostname
    }

    print_modules() {
        echo -e "\n\n Which module in this job."
        module list 2>&1
    }

    print_environment() {
        echo -e "\n\n what is the environment."
        printenv
    }

    print_working_folder() {
        cd \${PBS_O_WORKDIR}
        echo -e "my work folder is \$PWD\n\n"
        echo
    }

    run_job() {
        echo The job \$PBS_JOBNAME is running on \`cat \$PBS_NODEFILE\`.
        hostname > /home/users/$username/$log_dir/\${PBS_JOBID}_\${PBS_JOBNAME}.txt 2>&1
    }

    source run_mlflow.sh

    # Execute functions
    cd \$PBS_O_WORKDIR
    print_hostname
    print_modules
    print_environment
    print_working_folder
    run_job
    start_mlflow_server
    run_mlflow
EOF
}

create_pbs_script "$@"

create_pbs_script "$@"