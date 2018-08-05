#!/usr/bin/env bash
# 하나씩 또는 전체 예제 실행.
# runexamples.sh -h 를 통해 가능한 옵션 확인.
#   -h|--help          도움말을 보여주고 종료
#   -a|--all           모든 예제 실행. 기본값은 실행할 예제 번호 입력을 대기함. 
#   -n|--no-pauses     예제 사이에 대기하지 않음.(--all 옵션과 함께 사용)

help() {
  [ -n "$1" ] && echo $1
  cat <<-EOF1
usage: $0 [-h|--help] [-a|-all] [-n|--no-pauses]
where:
  -h|--help          Show help and quit
  -a|--all           Run all the examples. Default is to prompt for which one to run.
  -n|--no-pauses     Don't pause between examples (use with --all).
}
EOF1
}

# # 모든 예제를 실행하는 경우에만 표시됨.
#   각 예제가 실행되고 나면 몇몇 예제는 데이터 플롯이 포함 된 다이얼로그 팝업이 나타날
#   수 있다. 데이터 플롯 애플리케이션을 종료하고 다음 예제를 실행하면 된다. 
#   각 예제가 실행된 후에는 일시정지 된다. <return>을 눌러서 계속하거나 <ctrl-c>를 눌러서 종료하라.
banner() {
  if [ $all -eq 0 ]
  then
    cat <<-EOF2
=======================================================================

    deeplearning4j examples:

    Each example will be executed, then some of them will pop up a
    dialog with a data plot. Quit the data plot application to proceed
    to the next example.
EOF2
    if [ $pauses -eq 0 ]
    then
    cat <<-EOF2

    We'll pause after each one; hit <return> to continue or <ctrl-c>
    to quit.
EOF2
    fi
    cat <<-EOF3

=======================================================================
EOF3
  fi
}


let all=1
let pauses=0
while [ $# -ne 0 ]
do
  case $1 in
    -h|--h*)
      help
      exit 0
      ;;
    -a|--a*)
      let all=0
      ;;
    -n|--n*)
      let pauses=1
      ;;
    *)
      help "Unrecognized argument $1"
      exit 1
      ;;
  esac
  shift
done

# 대부분의 클래스 이름이 "Example"로 끝나지만 모두 그런것은 아니다. 
# Java 파일 중에서 "void main"을 찾으면 모든 클래스를 찾을 수 있다. 

dir=$PWD
cd dl4j-examples/src/main/java

find_examples() {
  # "main"을 검색해서 모든 Java 파일을 찾은 다음 '/'를 '.'으로 변경.
  # '.' 이후의 불필요한 공란 삭제 후 .java 확장자를 가지는 정규화된 클래스명 생성.
  find . -name '*.java' -exec grep -l 'void main' {} \; | \
    sed "s?/?.?g" | sed "s?^\.*\(.*\)\.java?\1?"
}


# 필드 구분기호는 \n.
# find_examples()의 결과를 string으로 변경하고 arr에 채우면 
# 클래스명 리스트인 arr가 생성 됨. 
eval "arr=($(find_examples))"

cd $dir


# Invoke with
#   NOOP=echo runexamples.sh
# to echo the command, but not run it.
runit() {
  echo; echo "====== $1"; echo
  $NOOP java -cp ./dl4j-examples/target/dl4j-examples-*-bin.jar "$1"
}

let which_one=0
if [ $all -ne 0 ]
then

  for index in "${!arr[@]}"   # arr의 index 반환
  do
    let i=$index+1
    echo "[$(printf "%2d" $i)] ${arr[$index]}"
  done
  printf "Enter a number for the example to run: "
  read which_one
  if [ -z "$which_one" ]
  then
    which_one=0
  elif [ $which_one = 'q' ]  # 'q'를 종료('quit')로 간주
  then
    exit 0
  elif [ $which_one -le 0 -o $which_one -gt ${#arr[@]} ]
  then
    echo "ERROR: Must enter a number between 1 and ${#arr[@]}."
    exit 1
  else
    let which_one=$which_one-1
  fi

  runit ${arr[$which_one]}

else

  banner

  ## arr 내에서 반복 
  for e in "${arr[@]}"
  do
    runit "$e"
    if [ $pauses -eq 0 ]
    then
      echo; echo -n "hit return to continue: "
      read toss
    fi
  done
fi
