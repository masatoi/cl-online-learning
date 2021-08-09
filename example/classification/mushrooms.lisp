;;; -*- coding:utf-8; mode:lisp -*-

;;; mushrooms: Multiclass classification
;; dataset:
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter mushrooms-dim 112)

(defparameter mushrooms-train (clol.utils:read-data "/home/wiz/datasets/mushrooms-train" mushrooms-dim))
(defparameter mushrooms-test (clol.utils:read-data "/home/wiz/datasets/mushrooms-test" mushrooms-dim))

(defparameter arow-learner (make-arow mushrooms-dim 10.0))
(loop repeat 10 do
  (train arow-learner mushrooms-train)
  (test arow-learner mushrooms-test))

;; Accuracy: 93.36158%, Correct: 1983, Total: 2124
