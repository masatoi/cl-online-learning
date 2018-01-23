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

(defparameter arow-learner (make-arow mushrooms-dim 10d0))
(loop repeat 10 do
  (train arow-learner mushrooms-train)
  (test arow-learner mushrooms-test))

;; Accuracy: 93.36158%, Correct: 1983, Total: 2124

(defparameter mushrooms-mul-train
  (let ((target (mapcar (lambda (datum)
                          (if (> (car datum) 0d0) 1 0))
                        mushrooms-train))
        (input (mapcar #'cdr mushrooms-train)))
    (mapcar #'cons target input)))

(defparameter mushrooms-mul-test
  (let ((target (mapcar (lambda (datum)
                          (if (> (car datum) 0d0) 1 0))
                        mushrooms-test))
        (input (mapcar #'cdr mushrooms-test)))
    (mapcar #'cons target input)))

(defparameter arow-mul-learner (make-one-vs-rest mushrooms-dim 2 'arow 10d0))
(loop repeat 10 do
  (train arow-mul-learner mushrooms-mul-train)
  (test arow-mul-learner mushrooms-mul-test))
