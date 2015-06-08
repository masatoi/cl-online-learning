;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning
  (:use :cl :hjs.util.vector)
  (:nicknames :cl-ol))

(in-package :cl-online-learning)

;;; Signum function
(defun sign (x)
  (if (> x 0d0) 1d0 -1d0))

;;; Decision boundary
(defun f (input weight bias)
  (+ (inner-product weight input) bias))

;;; Definition CLOS object
(defclass learner ()
  ((input-dimension :accessor input-dimension-of :initarg :input-dimension)
   (weight :accessor weight-of :initarg :weight)
   (bias :accessor bias-of :initarg :bias)))

;;; Prediction
(defmethod predict ((learner learner) input)
  (sign (f input (weight-of learner) (bias-of learner))))

;;; Testing with a test data list (1-pass)
(defmethod test ((learner learner) test-data)
  (let* ((len (length test-data))
	 (n-correct (count-if (lambda (datum)
				(= (predict learner (cdr datum)) (car datum)))
			      test-data))
	 (accuracy (* (/ n-correct len) 100.0)))
    (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" accuracy n-correct len)
    (values accuracy n-correct len)))

;;; Update inner parameters destructively (1step)
(defmethod update ((learner learner) input training-label) learner)

;;; Training with a training data list (1-pass)
(defmethod train ((learner learner) training-data)
  (dolist (datum training-data)
    (update learner (cdr datum) (car datum)))
  learner)

;;; Perceptron
(defclass perceptron (learner) ())

(defun make-perceptron (input-dimension)
  (check-type input-dimension integer)
  (make-instance 'perceptron
     :input-dimension input-dimension
     :weight (make-dvec input-dimension 0d0)
     :bias 0d0))

(defmethod update ((learner perceptron) input training-label)
  (if (<= (* training-label (f input (weight-of learner) (bias-of learner))) 0d0)
    (if (> training-label 0d0)
      (progn
	(v+ (weight-of learner) input (weight-of learner))
	(setf (bias-of learner) (+ (bias-of learner) 1d0)))
      (progn
	(v- (weight-of learner) input (weight-of learner))
	(setf (bias-of learner) (- (bias-of learner) 1d0)))))
  learner)

;;; Averaged Perceptron
(defclass averaged-perceptron (perceptron)
  ((data-size :accessor data-size-of :initarg :data-size)
   (averaged-weight :accessor averaged-weight-of :initarg :averaged-weight)
   (averaged-bias :accessor averaged-bias-of :initarg :averaged-bias)
   (counter :accessor counter-of :initarg :counter)
   (tmp-vec :accessor tmp-vec-of :initarg :tmp-vec)))

(defun make-averaged-perceptron (input-dimension data-size)
  (check-type input-dimension integer)
  (check-type data-size integer)
  (make-instance 'averaged-perceptron
     :input-dimension input-dimension
     :data-size data-size
     :weight (make-dvec input-dimension 0d0)
     :bias 0d0
     :averaged-weight (make-dvec input-dimension 0d0)
     :averaged-bias 0d0
     :counter 1
     :tmp-vec (make-dvec input-dimension 0d0)))

(defmethod update ((learner averaged-perceptron) input training-label)
  (if (<= (* training-label (f input (weight-of learner) (bias-of learner))) 0d0)
    (let ((average-factor (- 1d0 (/ (- (counter-of learner) 1d0) (data-size-of learner)))))
      (if (> training-label 0d0)
	(progn
	  (v+ (weight-of learner) input (weight-of learner))
	  (v+ (averaged-weight-of learner)
	      (v-scale input average-factor (tmp-vec-of learner)) (averaged-weight-of learner))
	  (setf (bias-of learner) (+ (bias-of learner) 1d0)
		(averaged-bias-of learner) (+ (averaged-bias-of learner) average-factor)))
	(progn
	  (v- (weight-of learner) input (weight-of learner))
	  (v- (averaged-weight-of learner)
	      (v-scale input average-factor (tmp-vec-of learner)) (averaged-weight-of learner))
	  (setf (bias-of learner) (- (bias-of learner) 1d0)
		(averaged-bias-of learner) (- (averaged-bias-of learner) average-factor))))))
  (incf (counter-of learner))
  learner)

(defmethod predict ((learner averaged-perceptron) input)
  (sign (f input (averaged-weight-of learner) (averaged-bias-of learner))))

;;; Linear SVM
(defclass svm (learner)
  ((learning-rate :accessor learning-rate-of :initarg :learning-rate)
   (regularization-parameter :accessor regularization-parameter-of :initarg :regularization-parameter)
   (tmp-vec :accessor tmp-vec-of :initarg :tmp-vec)))

(defun make-svm (input-dimension learning-rate regularization-parameter)
  (check-type input-dimension integer)
  (check-type learning-rate double-float)
  (check-type regularization-parameter double-float)
  (make-instance 'svm
     :input-dimension input-dimension
     :weight (make-dvec input-dimension 0d0)
     :bias 0d0
     :learning-rate learning-rate
     :regularization-parameter regularization-parameter
     :tmp-vec (make-dvec input-dimension 0d0)))

(defmethod update ((learner svm) input training-label)
  (let* ((update-p (<= (* training-label (f input (weight-of learner) (bias-of learner))) 1d0))
	 (tmp-weight
	  (if update-p
	    (v+ (weight-of learner)
		(v-scale input (* (learning-rate-of learner) training-label) (tmp-vec-of learner))
		(weight-of learner))
	    (weight-of learner)))
	 (tmp-bias (if update-p
		     (+ (bias-of learner) (* (learning-rate-of learner) training-label))
		     (bias-of learner)))
	 (coefficient (- 1d0 (* 2d0 (learning-rate-of learner) (regularization-parameter-of learner)))))
    (v-scale tmp-weight coefficient (weight-of learner))
    (setf (bias-of learner) (* tmp-bias coefficient)))
  learner)

;;; AROW
(defclass arow (learner)
  ((gamma :accessor gamma-of :initarg :gamma)
   (weight-confidence :accessor weight-confidence-of :initarg :weight-confidence)
   (bias-confidence :accessor bias-confidence-of :initarg :bias-confidence)
   (tmp-vec1 :accessor tmp-vec1-of :initarg :tmp-vec1)
   (tmp-vec2 :accessor tmp-vec2-of :initarg :tmp-vec2)))

(defun make-arow (input-dimension gamma)
  (check-type input-dimension integer)
  (check-type gamma double-float)
  (make-instance 'arow
     :input-dimension input-dimension
     :weight (make-dvec input-dimension 0d0)
     :bias 0d0
     :gamma gamma
     :weight-confidence (make-dvec input-dimension 1d0)
     :bias-confidence 1d0
     :tmp-vec1 (make-dvec input-dimension 0d0)
     :tmp-vec2 (make-dvec input-dimension 0d0)))

(defmethod update ((learner arow) input training-label)
  (let* ((loss (- 1d0 (* training-label (f input (weight-of learner) (bias-of learner))))))
    (if (> loss 0d0)
      (let* ((beta (/ 1d0 (+ (bias-confidence-of learner)
			     (inner-product (diagonal-matrix-multiplication (weight-confidence-of learner)
									    input
									    (tmp-vec1-of learner))
					    input)
			     (gamma-of learner))))
	     (alpha (* loss beta)))
	;; Update weight
	(v-scale (tmp-vec1-of learner) (* alpha training-label) (tmp-vec2-of learner))
	(v+ (weight-of learner) (tmp-vec2-of learner) (weight-of learner))
	;; Update bias
	(setf (bias-of learner) (+ (bias-of learner) (* alpha (bias-confidence-of learner) training-label)))
	;; Update weight-confidence
	(diagonal-matrix-multiplication (tmp-vec1-of learner) (tmp-vec1-of learner) (tmp-vec1-of learner))
	(v-scale (tmp-vec1-of learner) beta (tmp-vec1-of learner))
	(v- (weight-confidence-of learner) (tmp-vec1-of learner) (weight-confidence-of learner))
	;; Update bias-confidence
	(setf (bias-confidence-of learner)
	      (- (bias-confidence-of learner)
		 (* beta (bias-confidence-of learner)
		    (bias-confidence-of learner)))))))
  learner)
