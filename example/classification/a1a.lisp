;;; -*- coding:utf-8; mode:lisp -*-

;;; A1A: Binary classification, Dence data
;; dataset:
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter a1a-dim 123)
(defparameter a1a-train (read-data "/home/wiz/datasets/a1a" a1a-dim))
(defparameter a1a-test (read-data "/home/wiz/datasets/a1a.t" a1a-dim))

(defparameter perceptron-learner (make-perceptron a1a-dim))
(train perceptron-learner a1a-train)
(test perceptron-learner a1a-test)

(defparameter arow-learner (make-arow a1a-dim 10d0))
(train arow-learner a1a-train)
(test arow-learner a1a-test)

(defparameter scw-learner (make-scw  a1a-dim 0.9d0 0.1d0))
(train scw-learner a1a-train)
(test scw-learner a1a-test)

(defparameter sgd-learner (make-lr+sgd a1a-dim 0.00001d0 0.01d0))
(train sgd-learner a1a-train)
(test sgd-learner a1a-test)

(loop for C in '(0d0 0.00001d0 0.0001d0 0.001d0 0.01d0 0.1d0 1d0) do
  (defparameter sgd-learner (clol::make-lr+sgd a1a-dim C 0.01d0))
  (print (clol::sgd-C sgd-learner))
  (loop repeat 20 do
    (train sgd-learner a1a-train)
    (test  sgd-learner a1a-test)))

; α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10^-8
(defparameter adam-learner (make-lr+adam a1a-dim 0.000001d0 0.001d0 1.d-8 0.9d0 0.99d0))
(train adam-learner a1a-train)
(test adam-learner a1a-test)

(loop for C in '(0d0 0.00001d0 0.0001d0 0.001d0 0.01d0 0.1d0 1d0) do
  (defparameter adam-learner (make-lr+adam a1a-dim C 0.001d0 1.d-8 0.9d0 0.99d0))
  (format t "lr+adam-C: ~A~%" (clol::lr+adam-C adam-learner))
  (loop repeat 20 do
    (train adam-learner a1a-train)
    (test  adam-learner a1a-test)))

;;; Binary classification, Sparse data

(defparameter sparse-a1a-train (read-data "/home/wiz/tmp/a1a" a1a-dim :sparse-p t))
(defparameter sparse-a1a-test (read-data "/home/wiz/tmp/a1a.t" a1a-dim :sparse-p t))

(defparameter sparse-perceptron-learner (make-sparse-perceptron a1a-dim))
(train sparse-perceptron-learner sparse-a1a-train)
(test sparse-perceptron-learner sparse-a1a-test)

(defparameter sparse-arow-learner (make-sparse-arow a1a-dim 10d0))
(train sparse-arow-learner sparse-a1a-train)
(test sparse-arow-learner sparse-a1a-test)

(defparameter sparse-scw-learner (make-sparse-scw a1a-dim 0.9d0 0.1d0))
(train sparse-scw-learner sparse-a1a-train)
(sparse-scw-test sparse-scw-learner sparse-a1a-test)

(defparameter sparse-sgd-learner (make-sparse-lr+sgd a1a-dim 0.00001d0 0.01d0))
(train sparse-sgd-learner sparse-a1a-train)
(test sparse-sgd-learner sparse-a1a-test)

(defparameter sparse-adam-learner (make-sparse-lr+adam a1a-dim 0.000001d0 0.001d0 1.d-8 0.9d0 0.99d0))
(train sparse-adam-learner sparse-a1a-train)
(test sparse-adam-learner sparse-a1a-test)
