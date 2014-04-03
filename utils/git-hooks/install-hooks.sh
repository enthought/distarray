#!/usr/bin/env bash

DOTGIT=`git rev-parse --git-dir`
TOPLEVEL=`git rev-parse --show-toplevel`
TO=${DOTGIT}/hooks
FROM=${TOPLEVEL}/utils/git-hooks

ln -s ${FROM}/pre-commit ${TO}/pre-commit
