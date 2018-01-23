;;; -*- coding:utf-8; mode:lisp -*-

;;; gisette: Binary classification, Many features
;; dataset:
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#gisette

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter gisette-dim 5000)
(defparameter gisette-train (read-data "/home/wiz/datasets/gisette_scale" gisette-dim))
(defparameter gisette-test (read-data "/home/wiz/datasets/gisette_scale.t" gisette-dim))

(defparameter perceptron-learner (make-perceptron gisette-dim))
(loop repeat 20 do
  (train perceptron-learner gisette-train)
  (test perceptron-learner gisette-test))

(defparameter arow-learner (make-arow gisette-dim 10d0))
(loop repeat 20 do
  (train arow-learner gisette-train)
  (test arow-learner gisette-test))

(defparameter scw-learner (make-scw gisette-dim 0.9d0 0.1d0))
(loop repeat 20 do
  (train scw-learner gisette-train)
  (test scw-learner gisette-test))

(defparameter sgd-learner (make-lr+sgd gisette-dim 0.00001d0 0.01d0))
(loop repeat 20 do
  (train sgd-learner gisette-train)
  (test sgd-learner gisette-test))

(loop for C in '(0d0 0.00001d0 0.0001d0 0.001d0 0.01d0 0.1d0 1d0) do
  (defparameter sgd-learner (clol::make-lr+sgd gisette-dim C 0.01d0))
  (print (clol::sgd-C sgd-learner))
  (loop repeat 20 do
    (train sgd-learner gisette-train)
    (test  sgd-learner gisette-test)))

; α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10^-8
(defparameter adam-learner (make-lr+adam gisette-dim 0.000001d0 0.001d0 1.d-8 0.9d0 0.99d0))
(loop repeat 20 do
  (train adam-learner gisette-train)
  (test adam-learner gisette-test))

(clgp:plots
 (list
  '(90.2 97.7 96.7 96.8 97.0 95.2 95.3 94.4 97.3 92.2 97.5 93.4 96.2 94.8 96.5
        96.3 96.6 96.6 96.6 96.6)
  '(97.2 96.3 97.1 96.2 98.2 98.0 95.6 97.3 97.7 97.2 97.7 96.9 97.7 97.7 97.7
        97.7 97.7 97.7 97.7 97.7)
  '(96.5 96.4 95.7 97.5 95.8 97.3 96.9 95.4 94.4 97.8 97.1 97.1 96.7 97.7 97.3
        96.5 97.7 94.3 96.1 97.5)
  '(94.8 97.2 97.8 97.7 97.6 97.2 97.2 97.0 96.8 96.8 97.0 97.0 97.1 97.1 97.3
 97.3 97.6 97.6 97.5 97.3)
  '(95.3 97.2 97.2 94.8 97.5 96.3 97.1 97.0 96.6 97.7 97.1 97.3 97.5 97.7 97.7
    97.3 96.5 97.2 97.6 97.2))
 :title-list '("perceptron" "lr+sgd" "lr+adam" "arow" "scw")
 :x-label "epochs"
 :y-label "accuracy for gisette")
