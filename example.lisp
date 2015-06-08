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
			    (training-label (coerce (parse-integer (car d)) 'double-float)))
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
(let ((learner (make-perceptron a1a-dim)))
  (train learner a1a-train)
  (test  learner a1a-test))

;; Averaged Perceptron
(let ((learner (make-averaged-perceptron a1a-dim (length a1a-train))))
  (train learner a1a-train)
  (test  learner a1a-test))

;; Linear SVM
(let ((learner (make-svm a1a-dim 0.01d0 0.01d0))) ; learning-rate & regularization-parameter
  (train learner a1a-train)
  (test  learner a1a-test))

;; AROW
(let ((learner (make-arow a1a-dim 10d0))) ; gamma
  (train learner a1a-train)
  (test  learner a1a-test))

;;; a9a dataset

;; Fetch dataset
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
;; $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t

(defparameter a9a-dim 123)
(defparameter a9a-train (read-libsvm-data "/home/wiz/tmp/a9a" a9a-dim))
(defparameter a9a-test (read-libsvm-data "/home/wiz/tmp/a9a.t" a9a-dim))

;; Perceptron
(let ((learner (make-perceptron a9a-dim)))
  (train learner a9a-train)
  (test  learner a9a-test))

;; Averaged Perceptron
(let ((learner (make-averaged-perceptron a1a-dim (length a9a-train))))
  (train learner a9a-train)
  (test  learner a9a-test))

;; Linear SVM
(let ((learner (make-svm a9a-dim 0.01d0 0.01d0))) ; learning-rate & regularization-parameter
  (train learner a9a-train)
  (test  learner a9a-test))

;; AROW
(let ((learner (make-arow a9a-dim 10d0))) ; gamma
  (train learner a9a-train)
  (test  learner a9a-test))
