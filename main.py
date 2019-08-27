"""Main entry of DLKoASR."""

import os
import os.path
import logging
import argparse
import tensorflow as tf
import model
import utils
import utils.hangul


# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
# OOM 에러 떠서 어느 부분이 문제인지 찾아보려고 넣은건데, model.compile(~ , option = run_opts) 이렇게 쓰는건데, model.compile 못찾아서 보류..

parser = argparse.ArgumentParser()
parser.add_argument("--sess-dir", type=str,
                    help="Session directory", default = './ny')
# 근데, default 값으로 경로 지정, 터미널에다 python3 main.py --sess-dir ny 해서 돌리면 학습 시작.
# ny 폴더 생기고, 그안에 train, valid 폴더가 생김, 그리고 학습 돌리면 체크 포인트 기타 등등이 저장된다.
# 다시 학습을 돌리려고 할때, 이 폴더를 지워줘야 함.. 저장 되어있던것 때문에 train 모드 대신 test로 돌아가기 때문.

parser.add_argument("--config", type=str, default="configs/default.yml",
                    help="Configuration file.")
# config 폴더 안에 default 파일 없고, 대신 tmp.sh 있는데 무슨일..?
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s:%(lineno)d %(message)s")
log = logging.getLogger("DLKoASR")
# 로그 기록


if __name__ == "__main__":
  args = parser.parse_args()
  try:
    os.makedirs(args.sess_dir)  # 아까 위에서 말한 디렉토리를 만들어 주고자 하는 부분.
  except FileExistsError as e:
    if not os.path.isdir(args.sess_dir):
      raise e                   # 못만들면 에러 뜸.
  writers = {phase: tf.summary.FileWriter(os.path.join(args.sess_dir, phase))
             for phase in ["train", "valid"]}    # 위에 만든 디렉도리 안에 train, valid 하위 폴더 생성.
  utils.load_config(args.config)                 # utils 폴더의 __init__.py 파일의 load_config 함수 : 파라미터들의 기본값을 지정.

  model = model.KoSR(args.sess_dir)              # model 폴더의 __init__.py 파일의 class KoSR을 불러온다.
  with tf.device(model.device):                  # 학습 device 선택 (GPU)
    train_batch, valid_batch, test_batch = utils.load_batches()  # utils 폴더의 __init__.py 파일의 load_batches 함수

  with tf.Session(config=model.config) as sess:
    saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=1)
    try:
      saver.restore(sess, tf.train.latest_checkpoint(args.sess_dir))        # save the checkpoint
    except ValueError:
      log.info("==== TRAINING START ====")
      writers["train"].add_graph(sess.graph)
      model.train(sess, train_batch, valid_batch,
                  writers["train"], writers["valid"], saver )
      saver.save(sess, model.path)
    log.info("===== TESTING START ====")
    model.test(sess, test_batch)
