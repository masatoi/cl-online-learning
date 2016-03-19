;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :cl-ol.exam))

(in-package :cl-online-learning.examples)

;;; a1a dataset

;; Fetch dataset
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t

(defparameter a1a-dim 123)
(defparameter a1a-train (read-libsvm-data "/home/wiz/tmp/a1a" a1a-dim))
(defparameter a1a-test (read-libsvm-data "/home/wiz/tmp/a1a.t" a1a-dim))

;; Perceptron
(defparameter perceptron-learner (make-perceptron a1a-dim))
(defparameter perceptron-trainer (train perceptron-learner a1a-train))
(test  perceptron-learner a1a-test)

;; Averaged Perceptron
(defparameter averaged-perceptron-learner (make-averaged-perceptron a1a-dim (length a1a-train)))
(train averaged-perceptron-learner a1a-train)
(test  averaged-perceptron-learner a1a-test)

;; Linear SVM
(defparameter svm-learner (make-svm a1a-dim 0.01d0 0.01d0)) ; learning-rate, regularization-parameter
(train svm-learner a1a-train)
(test  svm-learner a1a-test)

;; AROW
(defparameter arow-learner (make-arow a1a-dim 10d0)) ; gamma
(train arow-learner a1a-train)
(test  arow-learner a1a-test)

;; SCW-I
(defparameter scw1-learner (make-scw1 a1a-dim 0.8d0 0.1d0)) ; eta, C
(train scw1-learner a1a-train)
(test  scw1-learner a1a-test)

;; SCW-II
(defparameter scw2-learner (make-scw2 a1a-dim 0.8d0 0.1d0)) ; eta, C
(train scw2-learner a1a-train)
(test  scw2-learner a1a-test)

;;; a9a dataset

;; Fetch dataset
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t

(defparameter a9a-dim 123)
(defparameter a9a-train (read-libsvm-data "/home/wiz/tmp/a9a" a9a-dim))
(defparameter a9a-test (read-libsvm-data "/home/wiz/tmp/a9a.t" a9a-dim))

;; Perceptron
(defparameter perceptron-learner (make-perceptron a9a-dim))
(train perceptron-learner a9a-train)
(test  perceptron-learner a9a-test)

;; Averaged Perceptron
(defparameter averaged-perceptron-learner (make-averaged-perceptron a9a-dim (length a9a-train)))
(train averaged-perceptron-learner a9a-train)
(test  averaged-perceptron-learner a9a-test)

;; Linear SVM
(defparameter svm-learner (make-svm a9a-dim 0.01d0 0.01d0)) ; learning-rate, regularization-parameter
(train svm-learner a9a-train)
(test  svm-learner a9a-test)

;; AROW
(defparameter arow-learner (make-arow a9a-dim 10d0)) ; gamma
(train arow-learner a9a-train)
(test  arow-learner a9a-test)

;; SCW-I
(defparameter scw1-learner (make-scw1 a9a-dim 0.8d0 0.1d0)) ; eta, C
(train scw1-learner a9a-train)
(test  scw1-learner a9a-test)

;; SCW-II
(defparameter scw2-learner (make-scw2 a9a-dim 0.8d0 0.1d0)) ; eta, C
(train scw2-learner a9a-train)
(test  scw2-learner a9a-test)

;;; ijcnn1

(defparameter ijcnn1-dim 22)
(defparameter ijcnn1-train (read-libsvm-data "/home/wiz/tmp/ijcnn1" ijcnn1-dim))
(defparameter ijcnn1-test (read-libsvm-data "/home/wiz/tmp/ijcnn1.t" ijcnn1-dim))

;; Perceptron
(defparameter perceptron-learner (make-perceptron ijcnn1-dim))
(train perceptron-learner ijcnn1-train)
(test  perceptron-learner ijcnn1-train)
(test  perceptron-learner ijcnn1-test)

;; Averaged Perceptron
(defparameter averaged-perceptron-learner (make-averaged-perceptron ijcnn1-dim (length ijcnn1-train)))
(train averaged-perceptron-learner ijcnn1-train)
;; (defparameter averaged-perceptron-interim (train-with-interim-test averaged-perceptron-learner ijcnn1-train ijcnn1-test 100))
(test  averaged-perceptron-learner ijcnn1-train)
(test  averaged-perceptron-learner ijcnn1-test)

;; Linear SVM
(defparameter svm-learner (make-svm ijcnn1-dim 0.01d0 5d0)) ; learning-rate, regularization-parameter
(train svm-learner ijcnn1-train)
(test  svm-learner ijcnn1-train)
(test  svm-learner ijcnn1-test)

;; AROW
(defparameter arow-learner (make-arow ijcnn1-dim 0.0005d0))
(train arow-learner ijcnn1-train)
;; (defparameter arow-interim (train-with-interim-test arow-learner ijcnn1-train ijcnn1-test 100))
(test  arow-learner ijcnn1-train)
(test  arow-learner ijcnn1-test)

;; SCW-I
(defparameter scw1-learner (make-scw1 ijcnn1-dim 0.65d0 0.5d0)) ; eta, C
(train scw1-learner ijcnn1-train)
;; (defparameter scw1-interim (train-with-interim-test scw1-learner ijcnn1-train ijcnn1-test 100))
(test  scw1-learner ijcnn1-train)
(test  scw1-learner ijcnn1-test)

;; SCW-II
(defparameter scw2-learner (make-scw2 ijcnn1-dim 0.065d0 1d0)) ; eta, C
(train scw2-learner ijcnn1-train)
(test  scw1-learner ijcnn1-train)
(test  scw2-learner ijcnn1-test)

(defparameter ijcnn1-train-shuffled (shuffle-vector (coerce ijcnn1-train 'simple-vector)))

;; AROW
(defparameter arow-learner (make-arow ijcnn1-dim 0.0005d0))
(train arow-learner ijcnn1-train-shuffled)
;; (defparameter arow-interim (train-with-interim-test arow-learner ijcnn1-train ijcnn1-test 100))
(test  arow-learner ijcnn1-train-shuffled)
(test  arow-learner ijcnn1-test)

;; SCW-I
(defparameter scw1-learner (make-scw1 ijcnn1-dim 0.65d0 0.5d0)) ; eta, C
(train scw1-learner ijcnn1-train-shuffled)
;; (defparameter scw1-interim (train-with-interim-test scw1-learner ijcnn1-train ijcnn1-test 100))
(test  scw1-learner ijcnn1-train)
(test  scw1-learner ijcnn1-test)

;; SCW-II
(defparameter scw2-learner (make-scw2 ijcnn1-dim 0.065d0 1d0)) ; eta, C
(train scw2-learner ijcnn1-train-shuffled)
(test  scw1-learner ijcnn1-train)
(test  scw2-learner ijcnn1-test)

;;; cod-rna

(defparameter cod-rna-dim 8)
(defparameter cod-rna-train (read-libsvm-data "/home/wiz/tmp/cod-rna" cod-rna-dim))
(defparameter cod-rna-test (read-libsvm-data "/home/wiz/tmp/cod-rna.t" cod-rna-dim))
(defparameter cod-rna-train-shuffled (shuffle-vector (coerce cod-rna-train 'simple-vector)))

;; AROW
(defparameter arow-learner (make-arow cod-rna-dim 0.000000000000000000000000001d0))
(train arow-learner cod-rna-train)
(test  arow-learner cod-rna-train)
(test  arow-learner cod-rna-test)

(defparameter arow-learner (make-arow cod-rna-dim 0.000000000000000000000000001d0))
(train arow-learner cod-rna-train-shuffled)
(test  arow-learner cod-rna-train)
(test  arow-learner cod-rna-test)

;; SCW1
(defparameter scw1-learner (make-scw1 cod-rna-dim 0.51d0 1d0))
(train scw1-learner cod-rna-train)
(test  scw1-learner cod-rna-train)
(test  scw1-learner cod-rna-test)

(defparameter scw1-learner (make-scw1 cod-rna-dim 0.85d0 1d0))
(train scw1-learner cod-rna-train-shuffled)
(test  scw1-learner cod-rna-train)
(test  scw1-learner cod-rna-test)

;;; Multiclass classifier

;; vectorize and shuffle iris data
(defparameter iris
  (shuffle-vector
   (coerce (read-libsvm-data-multiclass "/home/wiz/tmp/iris.scale" 4)
	   'simple-vector)))

(defparameter iris-train (loop for i from 0 to (1- 100) collect (svref iris i)))
(defparameter iris-test (loop for i from 100 to (1- 150) collect (svref iris i)))  

;; one-vs-rest

;; Averaged Perceptron
(defparameter mul-percep (make-one-vs-rest 4 3 'perceptron))
(train mul-percep iris-train)
(test mul-percep iris-test)

;; AROW
(defparameter mul-arow (make-one-vs-rest 4 3 'arow 0.1d0))
(train mul-arow iris-train)
(test mul-arow iris-test)

;; SCW-I
(defparameter mul-scw1 (make-one-vs-rest 4 3 'scw1 0.65d0 0.5d0))
(train mul-scw1 iris-train)
(test mul-scw1 iris-test)

;; one-vs-one
;; Averaged Perceptron
(defparameter mul-percep (make-one-vs-one 4 3 'perceptron))
(train mul-percep iris-train)
(test mul-percep iris-test)

;; AROW
(defparameter mul-arow (make-one-vs-one 4 3 'arow 0.1d0))
(train mul-arow iris-train)
(test mul-arow iris-test)

;; SCW-I
(defparameter mul-scw1 (make-one-vs-one 4 3 'scw1 0.65d0 0.5d0))
(train mul-scw1 iris-train)
(test mul-scw1 iris-test)

;;; MNIST
(defparameter mnist (read-libsvm-data-multiclass "/home/wiz/tmp/mnist.scale" 780))
(defparameter mnist+1 (mapcar (lambda (x)
				(cons (1+ (car x)) (cdr x))) mnist))

(defparameter mnist.t (read-libsvm-data-multiclass "/home/wiz/tmp/mnist.scale.t" 780))
(defparameter mnist.t+1 (mapcar (lambda (x)
				(cons (1+ (car x)) (cdr x))) mnist.t))

(defparameter mulc (make-one-vs-rest 780 10 'arow 10d0))
(train mulc mnist+1)
(test mulc mnist.t+1)

(defparameter mulc (make-one-vs-one 780 10 'arow 10d0))
(train mulc mnist+1)
(test mulc mnist.t+1)

(defparameter mulc (make-one-vs-one 780 10 'scw1 0.65d0 10d0))
(train mulc mnist+1)
(test mulc mnist.t+1)

(defparameter mulc (make-ecoc 780 10 100 'arow 10d0))
(train mulc mnist+1)
(test mulc mnist.t+1)

(loop repeat 10 do
  (defparameter mulc (make-ecoc 780 10 20 'perceptron))
  (train mulc mnist+1)
  (test mulc mnist.t+1))

(loop repeat 10 do
  (defparameter mulc (make-ecoc 780 10 20 'perceptron))
  (print (cl-ol::codeword-matrix-of mulc)))

(array-dimensions (cl-ol::codeword-matrix-of mulc))

(car mnist+1)

(loop for learner across (cl-ol::learners-vector-of mulc) collect
  (predict learner (cdar mnist+1)))

(print (cl-ol::codeword-matrix-of mulc))
(update mulc (cdar mnist+1) (caar mnist+1))
(predict mulc (cdar mnist+1))

(defparameter mulc (make-one-vs-rest 780 10 'perceptron))
(train mulc mnist+1)
(test mulc mnist.t+1)

