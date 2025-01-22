#!/usr/bin/env bash

if [[ -z "${JULIA_NUM_THREADS}" ]]; then
    export JULIA_NUM_THREADS="${OMP_NUM_THREADS:-1}"
fi

julia pi.jl "${@}"
