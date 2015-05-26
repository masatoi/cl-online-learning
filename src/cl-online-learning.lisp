;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning
  (:use :cl :hjs.util.vector)
  (:nicknames :cl-ol))

(in-package :cl-online-learning)

;;; 符号関数
(defun sign (x)
  (if (> x 0d0) 1d0 -1d0))

;;; 線形識別器の決定境界
(defun f (input weight bias)
  (+ (inner-product weight input) bias))

;;; 線形識別器の予測
(defun predict (input weight bias)
  (sign (f input weight bias)))

(defun test (test-data weight bias)
  (let ((len (length test-data))
	(n-correct (count-if (lambda (datum)
			       (= (predict (cdr datum) weight bias) (car datum)))
			     test-data)))
    (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" (* (/ n-correct len) 100.0) n-correct len)))

(defun v+-no-sideeffect (v1 v2)
  (let* ((len (length v1))
	 (result (make-array len :element-type 'double-float)))
    (loop for i from 0 to (1- len) do
      (setf (aref result i) (+ (aref v1 i) (aref v2 i))))
    result))

;;; 3.3 パーセプトロン
;; アルゴリズム3.1
;; 破壊的に変更されたweightと非破壊的に計算されたbiasを返す
(defun train-perceptron-1step (input weight bias training-label)
  (if (<= (* training-label (f input weight bias)) 0d0)
    (if (> training-label 0d0)
      (values (v+ weight input weight) (+ bias 1d0))
      (values (v- weight input weight) (- bias 1d0)))
    (values weight bias)))

(defun train-perceptron-all (training-data weight bias)
  (loop for datum in training-data do
    (setf bias (nth-value 1 (train-perceptron-1step (cdr datum) weight bias (car datum)))))
  (values weight bias))

(defun train-perceptron (training-data)
  (let ((weight (make-dvec (length (cdar training-data)) 0d0))
	(bias 0d0))
    (train-perceptron-all training-data weight bias)))

;;; 3.6 サポートベクトルマシン
;; アルゴリズム3.3 線形SVM + 確率的勾配法(SGD)
;; 破壊的に変更されたweightと非破壊的に計算されたbiasを返す
(defun train-svm-sgd-1step (input weight bias learning-rate regularization-parameter
			    training-label v-scale-result)
  (let* ((update-p (<= (* training-label (inner-product weight input)) 1d0))
	 (tmp-weight
	  (if update-p
	    (v+ weight (v-scale input (* learning-rate training-label) v-scale-result) weight)
	    weight))
	 (tmp-bias (if update-p (+ bias (* learning-rate training-label)) bias)))
    (values
     (v-scale tmp-weight (- 1d0 (* 2d0 learning-rate regularization-parameter)) weight)
     (* tmp-bias (- 1d0 (* 2d0 learning-rate regularization-parameter))))))

(defun train-svm-sgd-all (training-data weight bias learning-rate regularization-parameter v-scale-result)
  (loop for datum in training-data do
    (setf bias
	  (nth-value 1 (train-svm-sgd-1step (cdr datum) weight bias
					    learning-rate regularization-parameter
					    (car datum) v-scale-result))))
  (values weight bias))

(defun train-svm-sgd (training-data learning-rate regularization-parameter)
  (let ((weight (make-dvec (length (cdar training-data)) 0d0))
	(bias 0d0)
	(v-scale-result (make-dvec (length (cdar training-data)) 0d0)))
    (train-svm-sgd-all training-data weight bias learning-rate regularization-parameter v-scale-result)))
