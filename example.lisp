;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-ol)

;;; データを用意する
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
			    (training-label (coerce (parse-integer (car d)) 'double-float)))
		       (dolist (index-num-pair index-num-alist)
			 (setf (aref dv (1- (car index-num-pair))) (cadr index-num-pair)))
		       (read-loop (cons (cons training-label dv) data-list)))))))
	(read-loop data-list)))))

;;;;; a1aに対する訓練とテスト

;; $ cd /tmp
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t
(defparameter a1a-train (read-libsvm-data "/home/wiz/tmp/a1a" 123))
(defparameter a1a-test (read-libsvm-data "/home/wiz/tmp/a1a.t" 123))

;; パーセプトロン
(multiple-value-bind (weight bias)
    (train-perceptron a1a-train)
  (test a1a-test weight bias))

;; 線形SVM+SGD
(let ((learning-rate 0.01d0)
      (regularization-parameter 0.01d0))
  (multiple-value-bind (weight bias)
      (train-svm-sgd a1a-train learning-rate regularization-parameter)
    (test a1a-test weight bias)))

;; AROW
(multiple-value-bind (mu sigma mu0-sigma0-vec)
    (train-arow a1a-train 10d0)
  (declare (ignore sigma))
  (test a1a-test mu (aref mu0-sigma0-vec 0)))

;;;;; a9aに対する訓練とテスト

;; $ cd /tmp
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t
(defparameter a9a-train (read-libsvm-data "/home/wiz/tmp/a9a" 123))
(defparameter a9a-test (read-libsvm-data "/home/wiz/tmp/a9a.t" 123))

;; パーセプトロン
(multiple-value-bind (weight bias)
    (train-perceptron a9a-train)
  (test a9a-test weight bias))

;; 線形SVM+SGD
(let ((learning-rate 0.01d0)
      (regularization-parameter 0.001d0))
  (multiple-value-bind (weight bias)
      (train-svm-sgd a9a-train learning-rate regularization-parameter)
    (test a9a-test weight bias)))

;; AROW
(multiple-value-bind (mu sigma mu0-sigma0-vec)
    (train-arow a9a-train 10d0)
  (declare (ignore sigma))
  (test a9a-test mu (aref mu0-sigma0-vec 0)))
