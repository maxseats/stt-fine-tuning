#!/bin/bash

# TMUX 세션 이름
TMUX_SESSION="maxseats_testing"

# 데이터셋 별 datasetkey 확인
# aihubshell -mode l

# 폴더 경로 변수
DOWNLOAD_DIR="/mnt/a/maxseats/testing"

# 데이터셋 키 변수
DATASET_KEY="464"



### 다운로드 ###

# TMUX 세션 존재 여부 확인
if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
  echo "TMUX session '$TMUX_SESSION' already exists. Exiting."
  exit 1
fi

# TMUX 실행 및 다운로드 폴더로 이동
tmux new-session -d -s $TMUX_SESSION


# 다운로드 디렉토리가 존재하지 않으면 생성
if [ ! -d "$DOWNLOAD_DIR" ]; then
  echo "Directory $DOWNLOAD_DIR does not exist. Creating..."
  mkdir -p "$DOWNLOAD_DIR"
fi
tmux send-keys -t $TMUX_SESSION "cd $DOWNLOAD_DIR" C-m

# AI hub 데이터셋 다운로드 명령어 실행
tmux send-keys -t $TMUX_SESSION "aihubshell -mode d -datasetkey $DATASET_KEY -aihubid email@mail.com -aihubpw password" C-m

# TMUX 세션에 접속하려면 다음 명령어를 사용하세요.
echo "To attach to the tmux session, use: tmux attach -t $TMUX_SESSION"



### 압축 해제 ###