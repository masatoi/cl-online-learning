;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning
  (:use :cl :hjs.util.vector)
  (:nicknames :cl-ol))

(in-package :cl-online-learning)

;; defclass-simplified: Simplified definition of classes which similar to definition of structure.
(defmacro defclass$ (class-name superclass-list &body body)
  "Simplified definition of classes which similar to definition of structure.
 [Example]
  (defclass$ agent (superclass1 superclass2)
    currency
    position-list
    (position-upper-bound 1)
    log
    money-management-rule)
=> #<STANDARD-CLASS AGENT>"
  `(defclass ,class-name (,@superclass-list)
     ,(mapcar (lambda (slot)
		(let* ((slot-symbol (if (listp slot) (car slot) slot))
		       (slot-name (symbol-name slot-symbol))
		       (slot-initval (if (listp slot) (cadr slot) nil)))
		  (list slot-symbol
			:accessor (intern (concatenate 'string slot-name "-OF"))
			:initarg (intern slot-name :KEYWORD)
			:initform slot-initval)))
	      body)))

;;; Signum function
(defun sign (x)
  (if (> x 0d0) 1d0 -1d0))

;;; Decision boundary
(defun f (input weight bias)
  (+ (inner-product weight input) bias))

;;; Definition CLOS object
(defclass$ learner () input-dimension weight bias)

;;; Definition generic functions
(defgeneric predict (learner input))
(defgeneric test (learner test-data  &key quiet-p))
(defgeneric update (learner input training-label))
(defgeneric train (learner training-data))
(defgeneric train-with-interim-test (learner training-data test-data span))

;;; Prediction
(defmethod predict ((learner learner) input)
  (sign (f input (weight-of learner) (bias-of learner))))

;;; Testing with a test data list (1-pass)
(defmethod test ((learner learner) test-data &key (quiet-p nil))
  (let* ((len (length test-data))
	 (n-correct (count-if (lambda (datum)
				(= (predict learner (cdr datum)) (car datum)))
			      test-data))
	 (accuracy (* (/ n-correct len) 100.0)))
    (if (not quiet-p)
      (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" accuracy n-correct len))
    (values accuracy n-correct len)))

;;; Update inner parameters destructively (1step)
(defmethod update ((learner learner) input training-label) learner)

;;; Training with a training data list (1-pass)
(defmethod train ((learner learner) training-data)
  (dolist (datum training-data)
    (update learner (cdr datum) (car datum)))
  learner)

(defmethod train-with-interim-test ((learner learner) training-data test-data span)
  (let ((result nil))
    (loop for i from 0 to (1- (length training-data))
	  for datum in training-data
	  do
       (update learner (cdr datum) (car datum))
       (when (zerop (mod i span))
	 (push (test learner test-data :quiet-p t) result)))
    (nreverse result)))

;;; Perceptron
(defclass$ perceptron (learner))

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
(defclass$ averaged-perceptron (perceptron)
  data-size averaged-weight averaged-bias counter tmp-vec)

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
(defclass$ svm (learner) learning-rate regularization-parameter tmp-vec)

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
(defclass$ arow (learner) gamma sigma sigma0 tmp-vec1 tmp-vec2)

(defun make-arow (input-dimension gamma)
  (check-type input-dimension integer)
  (check-type gamma double-float)
  (make-instance 'arow
     :input-dimension input-dimension
     :weight (make-dvec input-dimension 0d0) ; mu
     :bias 0d0                               ; mu0
     :gamma gamma
     :sigma (make-dvec input-dimension 1d0)
     :sigma0 1d0
     :tmp-vec1 (make-dvec input-dimension 0d0)
     :tmp-vec2 (make-dvec input-dimension 0d0)))

(defmethod update ((learner arow) input training-label)
  (let* ((loss (- 1d0 (* training-label (f input (weight-of learner) (bias-of learner))))))
    (if (> loss 0d0)
      (let* ((beta (/ 1d0 (+ (sigma0-of learner)
			     (inner-product (diagonal-matrix-multiplication (sigma-of learner)
									    input
									    (tmp-vec1-of learner))
					    input)
			     (gamma-of learner))))
	     (alpha (* loss beta)))
	;; Update weight
	(v-scale (tmp-vec1-of learner) (* alpha training-label) (tmp-vec2-of learner))
	(v+ (weight-of learner) (tmp-vec2-of learner) (weight-of learner))
	;; Update bias
	(setf (bias-of learner) (+ (bias-of learner) (* alpha (sigma0-of learner) training-label)))
	;; Update sigma
	(diagonal-matrix-multiplication (tmp-vec1-of learner) (tmp-vec1-of learner) (tmp-vec1-of learner))
	(v-scale (tmp-vec1-of learner) beta (tmp-vec1-of learner))
	(v- (sigma-of learner) (tmp-vec1-of learner) (sigma-of learner))
	;; Update sigma0
	(setf (sigma0-of learner)
	      (- (sigma0-of learner)
		 (* beta (sigma0-of learner)
		    (sigma0-of learner)))))))
  learner)

;;; SCW-I

;; Approximation of error function
(defun inverse-erf (x)
  (let* ((a (/ (* 8d0 (- pi 3d0))
	       (* 3d0 pi (- 4d0 pi))))
	 (c2/pia (/ 2d0 pi a))
	 (ln1-x^2 (log (- 1d0 (* x x))))
	 (comp (+ c2/pia (/ ln1-x^2 2d0))))
  (* (sign x)
     (sqrt (- (sqrt (- (* comp comp) (/ ln1-x^2 a)))
	      comp)))))

(defun probit (p)
  (* (sqrt 2d0)
     (inverse-erf (- (* 2d0 p) 1d0))))

(defclass$ scw1 (learner)
  ;; Meta parameters
  eta C
  ;; Internal parameters
  phi psi zeta sigma sigma0
  tmp-vec1 tmp-vec2)

(defun make-scw1 (input-dimension eta C)
  (check-type input-dimension integer)
  (check-type eta double-float)
  (check-type C double-float)
  (assert (< 0d0 eta 1d0))
  (let* ((phi (probit eta))
	 (psi (+ 1d0 (/ (* phi phi) 2d0)))
	 (zeta (+ 1d0 (* phi phi))))
    (make-instance 'scw1
       :input-dimension input-dimension
       :weight (make-dvec input-dimension 0d0)
       :bias 0d0
       :eta eta
       :C C
       :phi phi
       :psi psi
       :zeta zeta
       :sigma (make-dvec input-dimension 1d0)
       :sigma0 1d0
       :tmp-vec1 (make-dvec input-dimension 0d0)
       :tmp-vec2 (make-dvec input-dimension 0d0))))

(defmethod update ((learner scw1) input training-label)
  (let* ((phi (phi-of learner))
	 (m (* training-label (f input (weight-of learner) (bias-of learner))))
	 (v (+ (sigma0-of learner)
	       (inner-product (diagonal-matrix-multiplication (sigma-of learner) input (tmp-vec1-of learner))
			      input)))
	 (loss (- (* phi (sqrt v)) m)))
    (if (> loss 0d0)
      (let* ((psi (psi-of learner))
	     (zeta (zeta-of learner))
	     (alpha (min (C-of learner)
			 (max 0d0
			      (- (sqrt (+ (/ (* m m phi phi phi phi) 4d0)
					  (* v phi phi zeta)))
				 (* m psi)))))
	     (u (let ((base (- (sqrt (+ (* alpha alpha v v phi phi) (* 4d0 v))) (* alpha v phi))))
		  (/ (* base base) 4d0)))
	     (beta (/ (* alpha phi)
		      (+ (sqrt u) (* v alpha phi)))))
	;; Update weight
	(v-scale (tmp-vec1-of learner) (* alpha training-label) (tmp-vec2-of learner))
	(v+ (weight-of learner) (tmp-vec2-of learner) (weight-of learner))
	;; Update bias
	(setf (bias-of learner) (+ (bias-of learner) (* alpha (sigma0-of learner) training-label)))
	;; Update sigma
	(diagonal-matrix-multiplication (tmp-vec1-of learner) (tmp-vec1-of learner) (tmp-vec1-of learner))
	(v-scale (tmp-vec1-of learner) beta (tmp-vec1-of learner))
	(v- (sigma-of learner) (tmp-vec1-of learner) (sigma-of learner))
	;; Update sigma0
	(setf (sigma0-of learner)
	      (- (sigma0-of learner)
		 (* beta (sigma0-of learner)
		    (sigma0-of learner)))))))
  learner)

;;; SCW-II

(defclass$ scw2 (learner)
  ;; Meta parameters
  eta C
  ;; Internal parameters
  phi sigma sigma0
  tmp-vec1 tmp-vec2)

(defun make-scw2 (input-dimension eta C)
  (check-type input-dimension integer)
  (check-type eta double-float)
  (check-type C double-float)
  (assert (< 0d0 eta 1d0))
  (let ((phi (probit eta)))
    (make-instance 'scw2
       :input-dimension input-dimension
       :weight (make-dvec input-dimension 0d0)
       :bias 0d0
       :eta eta
       :C C
       :phi phi
       :sigma (make-dvec input-dimension 1d0)
       :sigma0 1d0
       :tmp-vec1 (make-dvec input-dimension 0d0)
       :tmp-vec2 (make-dvec input-dimension 0d0))))

(defmethod update ((learner scw2) input training-label)
  (let* ((phi (phi-of learner))
	 (m (* training-label (f input (weight-of learner) (bias-of learner))))
	 (v (+ (sigma0-of learner)
	       (inner-product (diagonal-matrix-multiplication (sigma-of learner) input (tmp-vec1-of learner))
			      input)))
	 (loss (- (* phi (sqrt v)) m)))
    (if (> loss 0d0)
      (let* ((n (+ v (/ 1d0 (* 2d0 (C-of learner)))))
	     (gamma (* phi
		       (sqrt (+ (* phi phi m m v v)
				(* 4d0 n v (+ n (* v phi phi)))))))
	     (alpha (max 0d0
			 (/ (- gamma (+ (* 2d0 m n) (* phi phi m v)))
			    (* 2d0 (+ (* n n) (* n v phi phi))))))
	     (u (let ((base (- (sqrt (+ (* alpha alpha v v phi phi) (* 4d0 v))) (* alpha v phi))))
		  (/ (* base base) 4d0)))
	     (beta (/ (* alpha phi)
		      (+ (sqrt u) (* v alpha phi)))))
	;; Update weight
	(v-scale (tmp-vec1-of learner) (* alpha training-label) (tmp-vec2-of learner))
	(v+ (weight-of learner) (tmp-vec2-of learner) (weight-of learner))
	;; Update bias
	(setf (bias-of learner) (+ (bias-of learner) (* alpha (sigma0-of learner) training-label)))
	;; Update sigma
	(diagonal-matrix-multiplication (tmp-vec1-of learner) (tmp-vec1-of learner) (tmp-vec1-of learner))
	(v-scale (tmp-vec1-of learner) beta (tmp-vec1-of learner))
	(v- (sigma-of learner) (tmp-vec1-of learner) (sigma-of learner))
	;; Update sigma0
	(setf (sigma0-of learner)
	      (- (sigma0-of learner)
		 (* beta (sigma0-of learner)
		    (sigma0-of learner)))))))
  learner)
