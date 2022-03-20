;;; -*- coding:utf-8; mode:lisp -*-

;;; cod-rna: Binary classification
;; dataset: 
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#cod-rna

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

;; Dence data

;; of classes: 2
;; of data: 59,535 / 271617 (validation) / 157413 (unused/remaining)
;; of features: 8

(defparameter cod-rna-dim 8)
(defparameter cod-rna (read-data "/home/wiz/datasets/cod-rna" cod-rna-dim))
(defparameter cod-rna.t (read-data "/home/wiz/datasets/cod-rna.t" cod-rna-dim))

;; it seem require scaling

(defparameter cod-rna.scale (read-data "/home/wiz/datasets/cod-rna.scale" cod-rna-dim))
(defparameter cod-rna.t.scale (read-data "/home/wiz/datasets/cod-rna.t.scale" cod-rna-dim))

(defparameter cod-rna-arow (make-arow cod-rna-dim 10.0))
(time (loop repeat 20 do (train cod-rna-arow cod-rna)))
(test cod-rna-arow cod-rna.t)

(defparameter cod-rna-arow (make-arow cod-rna-dim 0.1))
(loop repeat 20 do
  (train cod-rna-arow cod-rna)
  (test cod-rna-arow cod-rna.t))

(defparameter cod-rna-sgd (make-lr+sgd cod-rna-dim 0.000000001 0.001))
(loop repeat 20 do
  (train cod-rna-sgd cod-rna)
  (test cod-rna-sgd cod-rna.t))

(defparameter cod-rna-adam (make-lr+adam cod-rna-dim 0.00000000000000000001 0.001 1.e-8 0.9 0.99))
(loop repeat 20 do
  (train cod-rna-adam cod-rna)
  (test cod-rna-adam cod-rna.t))

(defparameter cod-rna-scw (make-scw cod-rna-dim 0.9 0.1))
(loop repeat 20 do
  (train cod-rna-scw cod-rna)
  (test cod-rna-scw cod-rna.t))

;; Evaluation took:
;;   0.989 seconds of real time
;;   0.991686 seconds of total run time (0.987975 user, 0.003711 system)
;;   [ Run times consist of 0.038 seconds GC time, and 0.954 seconds non-GC time. ]
;;   100.30% CPU
;;   3,356,870,825 processor cycles
;;   1,508,573,152 bytes consed
  
;; Accuracy: 79.89706%, Correct: 217014, Total: 271617

;; wiz@prime:~/datasets$ arow_learn -i 100 cod-rna cod-rna.model.arow
;; Number of features: 8
;; Number of examples: 59535
;; Number of updates:  5903680
;; Done!
;; Time: 9.0629 sec.

;; wiz@prime:~/datasets$ arow_test cod-rna.t cod-rna.model.arow
;; Accuracy 75.503% (205079/271617)
;; (Answer, Predict): (t,p):24835 (t,n):180244 (f,p):834 (f,n):65704
;; Done!
;; Time: 0.6416 sec.

(defparameter cod-rna.sp (read-data "/home/wiz/datasets/cod-rna" cod-rna-dim :sparse-p t))
(defparameter cod-rna.t.sp (read-data "/home/wiz/datasets/cod-rna.t" cod-rna-dim :sparse-p t))

;; it seem require scaling => use libsvm scaling

(defparameter cod-rna-arow.sp (make-sparse-arow cod-rna-dim 10.0))
(time (loop repeat 20 do (train cod-rna-arow.sp cod-rna.sp)))
(test cod-rna-arow.sp cod-rna.t.sp)

;; Evaluation took:
;;   1.170 seconds of real time
;;   1.172760 seconds of total run time (1.172760 user, 0.000000 system)
;;   [ Run times consist of 0.040 seconds GC time, and 1.133 seconds non-GC time. ]
;;   100.26% CPU
;;   3,968,874,560 processor cycles
;;   1,508,573,184 bytes consed
