;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-online-learning)

;;; Read libsvm dataset
(defun read-libsvm-data (data-path data-dimension)
  (let ((data-list nil))
    (with-open-file (f data-path :direction :input)
      (labels ((read-loop (data-list)
		 (let ((read-data (read-line f nil nil)))
		   (if (null read-data)
		     (nreverse data-list)
		     (let* ((dv (make-array data-dimension :element-type 'double-float :initial-element 0d0))
			    (d (ppcre:split "\\s+" read-data))
			    (index-num-alist
			     (mapcar (lambda (index-num-pair-str)
				       (let ((index-num-pair (ppcre:split #\: index-num-pair-str)))
					 (list (parse-integer (car index-num-pair))
					       (coerce (parse-number:parse-number (cadr index-num-pair)) 'double-float))))
				     (cdr d)))
			    (training-label (coerce (parse-number:parse-number (car d)) 'double-float)))
		       (dolist (index-num-pair index-num-alist)
			 (setf (aref dv (1- (car index-num-pair))) (cadr index-num-pair)))
		       (read-loop (cons (cons training-label dv) data-list)))))))
	(read-loop data-list)))))

;;; a1a dataset

;; Fetch dataset
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t

(defparameter a1a-dim 123)
(defparameter a1a-train (read-libsvm-data "/home/wiz/tmp/a1a" a1a-dim))
(defparameter a1a-test (read-libsvm-data "/home/wiz/tmp/a1a.t" a1a-dim))

;; Perceptron
(defparameter perceptron-learner (make-perceptron a1a-dim))
(train perceptron-learner a1a-train)
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
(defparameter averaged-perceptron-interim (train-with-interim-test averaged-perceptron-learner ijcnn1-train ijcnn1-test 100))
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
(defparameter arow-interim (train-with-interim-test arow-learner ijcnn1-train ijcnn1-test 100))
(test  arow-learner ijcnn1-train)
(test  arow-learner ijcnn1-test)

;; SCW-I
(defparameter scw1-learner (make-scw1 ijcnn1-dim 0.65d0 0.5d0)) ; eta, C
(train scw1-learner ijcnn1-train)
(defparameter scw1-interim (train-with-interim-test scw1-learner ijcnn1-train ijcnn1-test 100))
(test  scw1-learner ijcnn1-train)
(test  scw1-learner ijcnn1-test)

;; SCW-II
(defparameter scw2-learner (make-scw2 ijcnn1-dim 0.065d0 1d0)) ; eta, C
(train scw2-learner ijcnn1-train)
(test  scw1-learner ijcnn1-train)
(test  scw2-learner ijcnn1-test)
