;;; -*- coding:utf-8; mode:lisp -*-

;;; search difference of efficiency struct and CLOS

(in-package :cl-user)
(defpackage :cl-online-learning
  (:use :cl :cl-online-learning.vector)
  (:nicknames :clol)
  (:export
   :train :test
   :make-perceptron :perceptron-update :perceptron-train :perceptron-predict :perceptron-test
   :make-arow :arow-update :arow-train :arow-predict :arow-test
   :make-scw :scw-update :scw-train :scw-predict :scw-test
   :make-sgd :sgd-update :sgd-train :sgd-predict :sgd-test
   :make-adam :adam-update :adam-train :adam-predict :adam-test
   :make-sparse-perceptron :sparse-perceptron-update :sparse-perceptron-train
   :sparse-perceptron-predict :sparse-perceptron-test
   :make-sparse-arow :sparse-arow-update :sparse-arow-train :sparse-arow-predict :sparse-arow-test
   :make-sparse-scw :sparse-scw-update :sparse-scw-train :sparse-scw-predict :sparse-scw-test
   :make-sparse-sgd :sparse-sgd-update :sparse-sgd-train :sparse-sgd-predict :sparse-sgd-test
   :make-one-vs-rest :one-vs-rest-update :one-vs-rest-train :one-vs-rest-predict :one-vs-rest-test
   :make-one-vs-one :one-vs-one-update :one-vs-one-train :one-vs-one-predict :one-vs-one-test))

(in-package :cl-online-learning)

;;; Utils

(defmacro catstr (str1 str2)
  `(concatenate 'string ,str1 ,str2))

;; Signum
(defmacro sign (x)
  `(if (> ,x 0d0) 1d0 -1d0))

;; Decision boundary
(defmacro f (input weight bias)
  `(+ (dot ,weight ,input) ,bias))

;; Decision boundary (For sparse input)
(defmacro sf (input weight bias)
  `(+ (ds-dot ,weight ,input) ,bias))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun sparse-symbol? (symbol)
    (let ((name (symbol-name symbol)))
      (and (> (length name) 7)
           (string= (subseq (symbol-name symbol) 0 7)
                    "SPARSE-")))))

;;; Define learner functions (update, train, predict and test) at once by only writing update body.
(defmacro define-learner (learner-type (learner input training-label) &body body)
  `(progn
     (defun ,(intern (catstr (symbol-name learner-type) "-UPDATE"))
         (,learner ,input ,training-label)
       ,@body
       ,learner)
     (defun ,(intern (catstr (symbol-name learner-type) "-TRAIN"))
         (learner training-data)
       (etypecase training-data
         (list (dolist (datum training-data)
                 (,(intern (catstr (symbol-name learner-type) "-UPDATE"))
                   learner (cdr datum) (car datum))))
         (vector (loop for datum across training-data do
           (,(intern (catstr (symbol-name learner-type) "-UPDATE"))
                     learner (cdr datum) (car datum)))))
       learner)
     (defun ,(intern (catstr (symbol-name learner-type) "-PREDICT"))
         (learner input)
       (sign (,(if (sparse-symbol? learner-type) 'sf 'f)
              input
              (,(intern (catstr (symbol-name learner-type) "-WEIGHT")) learner)
              (,(intern (catstr (symbol-name learner-type) "-BIAS")) learner))))
     (defun ,(intern (catstr (symbol-name learner-type) "-TEST"))
         (learner test-data &key (quiet-p nil))
       (let* ((len (length test-data))
              (n-correct (count-if (lambda (datum)
                                     (= (,(intern (catstr (symbol-name learner-type) "-PREDICT"))
                                          learner (cdr datum)) (car datum)))
                                   test-data))
              (accuracy (* (/ n-correct len) 100.0)))
         (if (not quiet-p)
           (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" accuracy n-correct len))
         (values accuracy n-correct len)))))

(defun train (learner training-data)
  (funcall (intern (catstr (symbol-name (type-of learner)) "-TRAIN")
                   :cl-online-learning)
           learner training-data))

(defun test (learner test-data)
  (funcall (intern (catstr (symbol-name (type-of learner)) "-TEST")
                   :cl-online-learning)
           learner test-data))

;;; Perceptron

(defstruct (perceptron (:constructor %make-perceptron)
                       (:print-object %print-perceptron))
  input-dimension weight bias)

(defun %print-perceptron (obj stream)
  (format stream "#S(PERCEPTRON~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A)"
          (perceptron-input-dimension obj)
          (let ((w (perceptron-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (perceptron-bias obj)))

(defun make-perceptron (input-dimension)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (%make-perceptron :input-dimension input-dimension
                    :weight (make-dvec input-dimension 0d0)
                    :bias 0d0))

(define-learner perceptron (learner input training-label)
  (if (<= (* training-label
             (f input (perceptron-weight learner) (perceptron-bias learner)))
          0d0)
    (if (> training-label 0d0)
      (progn
        (v+ (perceptron-weight learner) input (perceptron-weight learner))
        (setf (perceptron-bias learner) (+ (perceptron-bias learner) 1d0)))
      (progn
        (v- (perceptron-weight learner) input (perceptron-weight learner))
        (setf (perceptron-bias learner) (- (perceptron-bias learner) 1d0))))))

;;; AROW

(defstruct (arow (:constructor  %make-arow)
                 (:print-object %print-arow))
  input-dimension weight bias
  gamma sigma sigma0 tmp-vec1 tmp-vec2)

(defun %print-arow (obj stream)
  (format stream "#S(AROW~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A~%~T:GAMMA ~A~%~T:SIGMA ~A ...~%~T:SIGMA0 ~A)"
          (arow-input-dimension obj)
          (let ((w (arow-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (arow-bias obj)
          (arow-gamma obj)
          (let ((s (arow-sigma obj)))
            (if (> (length s) 10)
              (subseq s 0 10)
              s))
          (arow-sigma0 obj)))

(defun make-arow (input-dimension gamma)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type gamma double-float)
  (%make-arow :input-dimension input-dimension
              :weight (make-dvec input-dimension 0d0) ; mu
              :bias 0d0                               ; mu0
              :gamma gamma
              :sigma (make-dvec input-dimension 1d0)
              :sigma0 1d0
              :tmp-vec1 (make-dvec input-dimension 0d0)
              :tmp-vec2 (make-dvec input-dimension 0d0)))

(define-learner arow (learner input training-label)
  (let ((loss (- 1d0 (* training-label (f input (arow-weight learner) (arow-bias learner))))))
    (if (> loss 0d0)
      (let* ((beta (/ 1d0 (+ (arow-sigma0 learner)
			     (dot (v* (arow-sigma learner) input (arow-tmp-vec1 learner))
                                  input)
			     (arow-gamma learner))))
	     (alpha (* loss beta)))
	;; Update weight
	(v*n (arow-tmp-vec1 learner) (* alpha training-label) (arow-tmp-vec2 learner))
	(v+ (arow-weight learner) (arow-tmp-vec2 learner) (arow-weight learner))
	;; Update bias
	(setf (arow-bias learner) (+ (arow-bias learner) (* alpha (arow-sigma0 learner) training-label)))
	;; Update sigma
	(v* (arow-tmp-vec1 learner) (arow-tmp-vec1 learner) (arow-tmp-vec1 learner))
	(v*n (arow-tmp-vec1 learner) beta (arow-tmp-vec1 learner))
	(v- (arow-sigma learner) (arow-tmp-vec1 learner) (arow-sigma learner))
	;; Update sigma0
	(setf (arow-sigma0 learner)
	      (- (arow-sigma0 learner)
		 (* beta (arow-sigma0 learner)
		    (arow-sigma0 learner))))))))

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

(defstruct (scw (:constructor  %make-scw)
                (:print-object %print-scw))
  input-dimension weight bias
  eta C
  ;; Internal parameters
  phi psi zeta sigma sigma0
  tmp-vec1 tmp-vec2)

(defun %print-scw (obj stream)
  (format stream "#S(SCW~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A~%~T:ETA ~A~%~T:C ~A~%~T:PHI ~A~%~T:PSI ~A~%~T:ZETA ~A~%~T:SIGMA ~A ...~%~T:SIGMA0 ~A)"
          (scw-input-dimension obj)
          (let ((w (scw-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (scw-bias obj)
          (scw-eta obj)
          (scw-C obj)
          (scw-phi obj)
          (scw-psi obj)
          (scw-zeta obj)
          (let ((s (scw-sigma obj)))
            (if (> (length s) 10)
              (subseq s 0 10)
              s))
          (scw-sigma0 obj)))

(defun make-scw (input-dimension eta C)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type eta double-float)
  (check-type C double-float)
  (assert (< 0d0 eta 1d0))
  (let* ((phi (coerce (probit eta) 'double-float))
	 (psi (+ 1d0 (/ (* phi phi) 2d0)))
	 (zeta (+ 1d0 (* phi phi))))
    (%make-scw
     :input-dimension input-dimension
     :weight (make-dvec input-dimension 0d0)
     :bias 0d0  :eta eta  :C C
     :phi phi   :psi psi  :zeta zeta
     :sigma    (make-dvec input-dimension 1d0)
     :sigma0 1d0
     :tmp-vec1 (make-dvec input-dimension 0d0)
     :tmp-vec2 (make-dvec input-dimension 0d0))))

(define-learner scw (learner input training-label)
  (let* ((phi (scw-phi learner))
	 (m (* training-label (f input (scw-weight learner) (scw-bias learner))))
	 (v (+ (scw-sigma0 learner)
	       (dot (v* (scw-sigma learner) input (scw-tmp-vec1 learner))
                    input)))
	 (loss (- (* phi (sqrt v)) m)))
    (if (> loss 0d0)
      (let* ((psi (scw-psi learner))
	     (zeta (scw-zeta learner))
	     (alpha (min (scw-C learner)
			 (max 0d0
			      (- (sqrt (+ (/ (* m m phi phi phi phi) 4d0)
					  (* v phi phi zeta)))
				 (* m psi)))))
	     (u (let ((base (- (sqrt (+ (* alpha alpha v v phi phi) (* 4d0 v))) (* alpha v phi))))
		  (/ (* base base) 4d0)))
	     (beta (/ (* alpha phi)
		      (+ (sqrt u) (* v alpha phi)))))
	;; Update weight
	(v*n (scw-tmp-vec1 learner) (* alpha training-label) (scw-tmp-vec2 learner))
	(v+ (scw-weight learner) (scw-tmp-vec2 learner) (scw-weight learner))
	;; Update bias
	(setf (scw-bias learner) (+ (scw-bias learner) (* alpha (scw-sigma0 learner) training-label)))
	;; Update sigma
	(v* (scw-tmp-vec1 learner) (scw-tmp-vec1 learner) (scw-tmp-vec1 learner))
	(v*n (scw-tmp-vec1 learner) beta (scw-tmp-vec1 learner))
	(v- (scw-sigma learner) (scw-tmp-vec1 learner) (scw-sigma learner))
	;; Update sigma0
	(setf (scw-sigma0 learner)
	      (- (scw-sigma0 learner)
		 (* beta (scw-sigma0 learner)
		    (scw-sigma0 learner))))))))

;;; Logistic regression (L2 regularization)

(defun sigmoid (x)
  (declare (type double-float x)
           (optimize (speed 3) (safety 0)))
  (/ 1d0 (+ 1d0 (exp (- x)))))

;; (defun logistic-regression-gradient (training-label input-vector weight-vector C tmp-vec result)
;;   (v*n input-vector
;;        (* (- 1d0
;;              (sigmoid (* training-label
;;                          (dot input-vector weight-vector))))
;;           (- training-label))
;;        tmp-vec)
;;   (v*n weight-vector (* 2d0 C) result)
;;   (v+ tmp-vec result result)
;;   result)

;; (defun logistic-regression-bias-gradient (training-label bias C)
;;   (+ (* (- 1d0 (sigmoid (* training-label bias)))
;;         (- training-label))
;;      (* 2d0 C bias)))

(defun logistic-regression-gradient (training-label input-vector weight-vector bias C tmp-vec result)
  (declare (type double-float training-label bias C)
           (type (simple-array double-float) input-vector weight-vector tmp-vec result)
           (optimize (speed 3) (safety 0)))
  (let ((sigmoid-val (sigmoid (* training-label (f input-vector weight-vector bias)))))
    ;; set gradient-vector to result
    (v*n input-vector
         (* (- 1d0 sigmoid-val) (- training-label))
         tmp-vec)
    (v*n weight-vector (* 2d0 C) result)
    (v+ tmp-vec result result)
    ;; return g0
    (+ (* (- 1d0 sigmoid-val)
          (- training-label))
       (* 2d0 C bias))))

(defstruct (sgd (:constructor %make-sgd))
  input-dimension weight bias
  ;; meta parameters
  C eta g tmp-vec)

(defun make-sgd (input-dimension C eta)
  (%make-sgd
   :input-dimension input-dimension
   :weight (make-dvec input-dimension 0d0)
   :bias 0d0
   :C C
   :eta eta
   :g (make-dvec input-dimension 0d0)
   :tmp-vec (make-dvec input-dimension 0d0)))

(define-learner sgd (learner input training-label)
  ;; calc g (gradient)
  (let ((g0 (logistic-regression-gradient training-label input
                                          (sgd-weight learner) (sgd-bias learner)
                                          (sgd-C learner) (sgd-tmp-vec learner) (sgd-g learner))))
    (v*n (sgd-g learner) (sgd-eta learner) (sgd-g learner))
    (v- (sgd-weight learner) (sgd-g learner) (sgd-weight learner))

    (setf (sgd-bias learner)
          (- (sgd-bias learner) (* (sgd-eta learner) g0)))))

;; Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980)
(defstruct (adam (:constructor %make-adam)
                 (:print-object %print-adam))
  input-dimension weight bias
  ;; meta parameters
  C alpha epsilon beta1 beta2
  ;; internal parameters
  g m v m0 v0 beta1^t beta2^t tmp-vec)

(defun %print-adam (obj stream)
  (format stream "#S(ADAM~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A)"
          (adam-input-dimension obj)
          (let ((w (adam-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (adam-bias obj)))

(defun make-adam (input-dimension C alpha epsilon beta1 beta2)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type C double-float)
  (check-type alpha double-float)
  (check-type epsilon double-float)
  (check-type beta1 double-float)
  (check-type beta2 double-float)
  (assert (< 0d0 alpha))
  (assert (and (<= 0d0 beta1) (< beta1 1d0)))
  (assert (and (<= 0d0 beta2) (< beta2 1d0)))
  (%make-adam
   :input-dimension input-dimension
   :weight (make-dvec input-dimension 0d0)
   :bias 0d0
   :C C
   :alpha alpha
   :epsilon epsilon
   :beta1 beta1
   :beta2 beta2
   :g (make-dvec input-dimension 0d0)
   :m (make-dvec input-dimension 0d0)
   :v (make-dvec input-dimension 0d0)
   :m0 0d0
   :v0 0d0
   :beta1^t beta1
   :beta2^t beta2
   :tmp-vec (make-dvec input-dimension 0d0)))

(define-learner adam (learner input training-label)
  ;; calc g (gradient)
  (let ((g0 (logistic-regression-gradient training-label input
                                          (adam-weight learner) (adam-bias learner)
                                          (adam-C learner) (adam-tmp-vec learner) (adam-g learner))))
    
    ;; update m_t from m_t-1
    (v*n (adam-m learner) (adam-beta1 learner) (adam-m learner))
    (v*n (adam-g learner) (- 1d0 (adam-beta1 learner)) (adam-tmp-vec learner))
    (v+ (adam-m learner) (adam-tmp-vec learner) (adam-m learner))

    ;; update m0
    (setf (adam-m0 learner)
          (+ (* (adam-beta1 learner) (adam-m0 learner))
             (* (- 1d0 (adam-beta1 learner)) g0)))

    ;; calc g^2 (gradient^2)
    (v* (adam-g learner) (adam-g learner) (adam-g learner))

    ;; update v_t from v_t-1
    (v*n (adam-v learner) (adam-beta2 learner) (adam-v learner))
    (v*n (adam-g learner) (- 1d0 (adam-beta2 learner)) (adam-tmp-vec learner))
    (v+ (adam-v learner) (adam-tmp-vec learner) (adam-v learner))

    ;; update v0
    (setf (adam-v0 learner)
          (+ (* (adam-beta2 learner) (adam-v0 learner))
             (* (- 1d0 (adam-beta2 learner)) (* g0 g0))))

    ;; update weight
    (let* ((epsilon-coefficient (sqrt (- 1d0 (adam-beta2^t learner))))
           (epsilon^ (* epsilon-coefficient (adam-epsilon learner)))
           (alpha_t (* (adam-alpha learner)
                       (/ epsilon-coefficient
                          (- 1d0 (adam-beta1^t learner))))))
      (v-sqrt (adam-v learner) (adam-tmp-vec learner))
      (v+n (adam-tmp-vec learner) epsilon^ (adam-tmp-vec learner))
      (v/ (adam-m learner) (adam-tmp-vec learner) (adam-tmp-vec learner))
      (v*n (adam-tmp-vec learner) alpha_t (adam-tmp-vec learner))
      (v- (adam-weight learner) (adam-tmp-vec learner) (adam-weight learner))

      ;; update bias
      (setf (adam-bias learner)
            (- (adam-bias learner)
               (* alpha_t (/ (adam-m0 learner)
                             (+ (sqrt (adam-v0 learner)) epsilon^))))))
    
    ;; update beta1^2 and beta2^2
    (setf (adam-beta1^t learner) (* (adam-beta1 learner) (adam-beta1^t learner))
          (adam-beta2^t learner) (* (adam-beta2 learner) (adam-beta2^t learner)))))

;;;; Sparse version learners ;;;;

;;; Sparse Perceptron

(defstruct (sparse-perceptron (:constructor %make-sparse-perceptron)
                              (:print-object %print-sparse-perceptron))
  input-dimension weight bias)

(defun %print-sparse-perceptron (obj stream)
  (format stream "#S(SPARSE-PERCEPTRON~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A)"
          (sparse-perceptron-input-dimension obj)
          (let ((w (sparse-perceptron-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (sparse-perceptron-bias obj)))

(defun make-sparse-perceptron (input-dimension)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (%make-sparse-perceptron :input-dimension input-dimension
                           :weight (make-dvec input-dimension 0d0)
                           :bias 0d0))

(define-learner sparse-perceptron (learner input training-label)
  (if (<= (* training-label (sf input
                                (sparse-perceptron-weight learner)
                                (sparse-perceptron-bias   learner))) 0d0)
    (if (> training-label 0d0)
      (progn
        (ds-v+ (sparse-perceptron-weight learner) input (sparse-perceptron-weight learner))
        (setf (sparse-perceptron-bias learner) (+ (sparse-perceptron-bias learner) 1d0)))
      (progn
        (ds-v- (sparse-perceptron-weight learner) input (sparse-perceptron-weight learner))
        (setf (sparse-perceptron-bias learner) (- (sparse-perceptron-bias learner) 1d0))))))

;;; Sparse AROW

(defstruct (sparse-arow (:constructor  %make-sparse-arow)
                        (:print-object %print-sparse-arow))
  input-dimension weight bias
  gamma sigma sigma0 tmp-vec1 tmp-vec2)

(defun %print-sparse-arow (obj stream)
  (format stream "#S(SPARSE-AROW~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A~%~T:GAMMA ~A~%~T:SIGMA ~A ...~%~T:SIGMA0 ~A)"
          (sparse-arow-input-dimension obj)
          (let ((w (sparse-arow-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (sparse-arow-bias obj)
          (sparse-arow-gamma obj)
          (let ((s (sparse-arow-sigma obj)))
            (if (> (length s) 10)
              (subseq s 0 10)
              s))
          (sparse-arow-sigma0 obj)))

(defun make-sparse-arow (input-dimension gamma)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type gamma double-float)
  (%make-sparse-arow :input-dimension input-dimension
                     :weight (make-dvec input-dimension 0d0) ; mu
                     :bias 0d0                               ; mu0
                     :gamma gamma
                     :sigma (make-dvec input-dimension 1d0)
                     :sigma0 1d0
                     :tmp-vec1 (make-dvec input-dimension 0d0)
                     :tmp-vec2 (make-dvec input-dimension 0d0)))

(define-learner sparse-arow (learner input training-label)
  (let ((index-vector (sparse-vector-index-vector input))
        (loss (- 1d0 (* training-label
                        (sf input (sparse-arow-weight learner) (sparse-arow-bias learner))))))
    (if (> loss 0d0)
      (let* ((beta (/ 1d0 (+ (sparse-arow-sigma0 learner)
			     (ds-dot (ds-v* (sparse-arow-sigma learner)
                                            input
                                            (sparse-arow-tmp-vec1 learner))
                                     input)
			     (sparse-arow-gamma learner))))
	     (alpha (* loss beta)))
	;; Update weight
	(ps-v*n (sparse-arow-tmp-vec1 learner)
                (* alpha training-label)
                index-vector
                (sparse-arow-tmp-vec2 learner))
	(dps-v+ (sparse-arow-weight learner)
                (sparse-arow-tmp-vec2 learner)
                index-vector
                (sparse-arow-weight learner))
	;; Update bias
	(setf (sparse-arow-bias learner) (+ (sparse-arow-bias learner)
                                            (* alpha (sparse-arow-sigma0 learner) training-label)))
	;; Update sigma
	(dps-v* (sparse-arow-tmp-vec1 learner)
                (sparse-arow-tmp-vec1 learner)
                index-vector
                (sparse-arow-tmp-vec1 learner))
	(ps-v*n (sparse-arow-tmp-vec1 learner)
                beta
                index-vector
                (sparse-arow-tmp-vec1 learner))
	(dps-v- (sparse-arow-sigma learner)
                (sparse-arow-tmp-vec1 learner)
                index-vector
                (sparse-arow-sigma learner))
	;; Update sigma0
	(setf (sparse-arow-sigma0 learner)
	      (- (sparse-arow-sigma0 learner)
		 (* beta (sparse-arow-sigma0 learner)
		    (sparse-arow-sigma0 learner))))))))

;;; Sparse SCW-I

(defstruct (sparse-scw (:constructor  %make-sparse-scw)
                       (:print-object %print-sparse-scw))
  input-dimension weight bias
  eta C
  ;; Internal parameters
  phi psi zeta sigma sigma0
  tmp-vec1 tmp-vec2)

(defun %print-sparse-scw (obj stream)
  (format stream "#S(SPARSE-SCW~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A~%~T:ETA ~A~%~T:C ~A~%~T:PHI ~A~%~T:PSI ~A~%~T:ZETA ~A~%~T:SIGMA ~A ...~%~T:SIGMA0 ~A)"
          (sparse-scw-input-dimension obj)
          (let ((w (sparse-scw-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (sparse-scw-bias obj)
          (sparse-scw-eta obj)
          (sparse-scw-C obj)
          (sparse-scw-phi obj)
          (sparse-scw-psi obj)
          (sparse-scw-zeta obj)
          (let ((s (sparse-scw-sigma obj)))
            (if (> (length s) 10)
              (subseq s 0 10)
              s))
          (sparse-scw-sigma0 obj)))

(defun make-sparse-scw (input-dimension eta C)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type eta double-float)
  (check-type C double-float)
  (assert (< 0d0 eta 1d0))
  (let* ((phi (coerce (probit eta) 'double-float))
	 (psi (+ 1d0 (/ (* phi phi) 2d0)))
	 (zeta (+ 1d0 (* phi phi))))
    (%make-sparse-scw
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

(define-learner sparse-scw (learner input training-label)
  (let* ((index-vector (sparse-vector-index-vector input))
         (phi (sparse-scw-phi learner))
         (m (* training-label (sf input (sparse-scw-weight learner) (sparse-scw-bias learner))))
         (v (+ (sparse-scw-sigma0 learner)
               (ds-dot (ds-v* (sparse-scw-sigma learner)
                              input
                              (sparse-scw-tmp-vec1 learner))
                       input)))
         (loss (- (* phi (sqrt v)) m)))
    (if (> loss 0d0)
      (let* ((psi (sparse-scw-psi learner))
             (zeta (sparse-scw-zeta learner))
             (alpha (min (sparse-scw-C learner)
                         (max 0d0
                              (- (sqrt (+ (/ (* m m phi phi phi phi) 4d0)
                                          (* v phi phi zeta)))
                                 (* m psi)))))
             (u (let ((base (- (sqrt (+ (* alpha alpha v v phi phi) (* 4d0 v))) (* alpha v phi))))
                  (/ (* base base) 4d0)))
             (beta (/ (* alpha phi)
                      (+ (sqrt u) (* v alpha phi)))))
        ;; Update weight
        (ps-v*n (sparse-scw-tmp-vec1 learner) (* alpha training-label) index-vector (sparse-scw-tmp-vec2 learner))
        (dps-v+ (sparse-scw-weight learner) (sparse-scw-tmp-vec2 learner) index-vector (sparse-scw-weight learner))
        ;; Update bias
        (setf (sparse-scw-bias learner)
              (+ (sparse-scw-bias learner) (* alpha (sparse-scw-sigma0 learner) training-label)))
        ;; Update sigma
        (dps-v* (sparse-scw-tmp-vec1 learner) (sparse-scw-tmp-vec1 learner) index-vector (sparse-scw-tmp-vec1 learner))
        (ps-v*n (sparse-scw-tmp-vec1 learner) beta index-vector (sparse-scw-tmp-vec1 learner))
        (dps-v- (sparse-scw-sigma learner) (sparse-scw-tmp-vec1 learner) index-vector (sparse-scw-sigma learner))
        ;; Update sigma0
        (setf (sparse-scw-sigma0 learner)
              (- (sparse-scw-sigma0 learner)
                 (* beta (sparse-scw-sigma0 learner)
                    (sparse-scw-sigma0 learner))))))))


;;; Logistic regression model (Sparse)

;; tmp-vec is pseudosparse-vector,

(defun logistic-regression-gradient-sparse
    (training-label input-vector weight-vector bias C tmp-vec result)
  (declare (type double-float training-label bias C)
           (type (simple-array double-float) weight-vector tmp-vec result)
           (optimize (speed 3) (safety 0)))
  (let ((sigmoid-val (sigmoid (* training-label (sf input-vector weight-vector bias)))))
    (sps-v*n input-vector
             (* (- 1d0 sigmoid-val) (- training-label))
             tmp-vec)
    (v*n weight-vector (* 2d0 C) result)
    (v+ tmp-vec result result)
    ;; return g0
    (+ (* (- 1d0 sigmoid-val)
          (- training-label))
       (* 2d0 C bias))))

;;; Sparse lr+sgd
(defstruct (sparse-sgd (:constructor %make-sparse-sgd))
  input-dimension weight bias
  ;; meta parameters
  C eta g tmp-vec)

(defun make-sparse-sgd (input-dimension C eta)
  (%make-sparse-sgd
   :input-dimension input-dimension
   :weight (make-dvec input-dimension 0d0)
   :bias 0d0
   :C C
   :eta eta
   :g (make-dvec input-dimension 0d0)
   :tmp-vec (make-dvec input-dimension 0d0)))

(define-learner sparse-sgd (learner input training-label)
  ;; calc g (gradient)
  (let ((g0 (logistic-regression-gradient-sparse
             training-label input
             (sparse-sgd-weight learner) (sparse-sgd-bias learner)
             (sparse-sgd-C learner) (sparse-sgd-tmp-vec learner) (sparse-sgd-g learner))))
    (v*n (sparse-sgd-g learner) (sparse-sgd-eta learner) (sparse-sgd-g learner))
    (v- (sparse-sgd-weight learner) (sparse-sgd-g learner) (sparse-sgd-weight learner))

    (setf (sparse-sgd-bias learner)
          (- (sparse-sgd-bias learner) (* (sparse-sgd-eta learner) g0)))))

;;;; Multiclass classifiers ;;;;

(defmacro define-multi-class-learner-train/test-functions (learner-type)
  `(progn
     (defun ,(intern (catstr (symbol-name learner-type) "-TRAIN"))
         (learner training-data)
       (etypecase training-data
         (list (dolist (datum training-data)
                 (,(intern (catstr (symbol-name learner-type) "-UPDATE"))
                   learner (cdr datum) (car datum))))
         (vector (loop for datum across training-data do
           (,(intern (catstr (symbol-name learner-type) "-UPDATE"))
                     learner (cdr datum) (car datum)))))
       learner)
     
     (defun ,(intern (catstr (symbol-name learner-type) "-TEST"))
         (learner test-data &key (quiet-p nil))
       (let* ((len (length test-data))
              (n-correct (count-if (lambda (datum)
                                     (= (,(intern (catstr (symbol-name learner-type) "-PREDICT"))
                                          learner (cdr datum)) (car datum)))
                                   test-data))
              (accuracy (* (/ n-correct len) 100.0)))
         (if (not quiet-p)
           (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" accuracy n-correct len))
         (values accuracy n-correct len)))))

;;; one-vs-rest

(defmacro function-by-name (name-string)
  `(symbol-function (intern ,name-string :cl-online-learning)))

(defstruct (one-vs-rest (:constructor  %make-one-vs-rest)
                        (:print-object %print-one-vs-rest))
  input-dimension n-class learners-vector
  learner-weight learner-bias learner-update learner-activate)

(defun %print-one-vs-rest (obj stream)
  (format stream "#S(ONE-VS-REST~%~T:INPUT-DIMENSION ~A~%~T:N-CLASS ~A~%~T:LEARNERS-VECTOR #(~A ...)~%~T:N-LEARNERS: ~A)"
          (one-vs-rest-input-dimension obj)
          (one-vs-rest-n-class obj)
          (if (vectorp (one-vs-rest-learners-vector obj))
            (type-of (aref (one-vs-rest-learners-vector obj) 0)))
          (if (vectorp (one-vs-rest-learners-vector obj))
            (length (one-vs-rest-learners-vector obj)))))

(defun make-one-vs-rest (input-dimension n-class learner-type &rest learner-params)
  (check-type input-dimension integer)
  (check-type n-class integer)
  (assert (> input-dimension 0))
  (assert (> n-class 2))
  (let ((mulc (%make-one-vs-rest
               :input-dimension input-dimension
               :n-class n-class
               :learners-vector (make-array n-class)
               :learner-weight (function-by-name (catstr (symbol-name learner-type) "-WEIGHT"))
               :learner-bias   (function-by-name (catstr (symbol-name learner-type) "-BIAS"))
               :learner-update (function-by-name (catstr (symbol-name learner-type) "-UPDATE"))
               :learner-activate (if (sparse-symbol? learner-type)
                                   (lambda (input weight bias)
                                     (+ (ds-dot weight input) bias))
                                   (lambda (input weight bias)
                                     (+ (dot weight input) bias))))))
    (loop for i from 0 to (1- n-class) do
      (setf (aref (one-vs-rest-learners-vector mulc) i)
            (apply (function-by-name (catstr "MAKE-" (symbol-name learner-type)))
                   (cons input-dimension learner-params))))
    mulc))

(defun one-vs-rest-predict (mulc input)
  (let ((max-f most-negative-double-float)
	(max-i 0))
    (loop for i from 0 to (1- (one-vs-rest-n-class mulc)) do
      (let* ((learner (svref (one-vs-rest-learners-vector mulc) i))
	     (learner-f (funcall (one-vs-rest-learner-activate mulc)
                                 input
                                 (funcall (one-vs-rest-learner-weight mulc) learner)
                                 (funcall (one-vs-rest-learner-bias mulc)   learner))))
	(if (> learner-f max-f)
	  (setf max-f learner-f
		max-i i))))
    max-i))

;; training-label should be integer (0 ... K-1)
(defun one-vs-rest-update (mulc input training-label)
  (loop for i from 0 to (1- (one-vs-rest-n-class mulc)) do
    (if (= i training-label)
      (funcall (one-vs-rest-learner-update mulc)
               (svref (one-vs-rest-learners-vector mulc) i) input 1d0)
      (funcall (one-vs-rest-learner-update mulc)
               (svref (one-vs-rest-learners-vector mulc) i) input -1d0))))

(define-multi-class-learner-train/test-functions one-vs-rest)

;;; one-vs-one

(defstruct (one-vs-one (:constructor  %make-one-vs-one)
                       (:print-object %print-one-vs-one))
  input-dimension n-class learners-vector
  learner-update learner-predict)

(defun %print-one-vs-one (obj stream)
  (format stream "#S(ONE-VS-ONE~%~T:INPUT-DIMENSION ~A~%~T:N-CLASS ~A~%~T:LEARNERS-VECTOR #(~A ...)~%~T:N-LEARNERS: ~A)"
          (one-vs-one-input-dimension obj)
          (one-vs-one-n-class obj)
          (if (vectorp (one-vs-one-learners-vector obj))
            (type-of (aref (one-vs-one-learners-vector obj) 0)))
          (if (vectorp (one-vs-one-learners-vector obj))
            (length (one-vs-one-learners-vector obj)))))

(defun make-one-vs-one (input-dimension n-class learner-type &rest learner-params)
  (check-type input-dimension integer)
  (check-type n-class integer)
  (assert (> input-dimension 0))
  (assert (> n-class 2))
  (let* ((n-learner (/ (* n-class (1- n-class)) 2))
	 (mulc (%make-one-vs-one
                :input-dimension input-dimension
                :n-class n-class
                :learners-vector (make-array n-learner)
                :learner-update (function-by-name (catstr (symbol-name learner-type) "-UPDATE"))
                :learner-predict (function-by-name (catstr (symbol-name learner-type) "-PREDICT")))))
    (loop for i from 0 to (1- n-learner) do
      (setf (aref (one-vs-one-learners-vector mulc) i)
            (apply (function-by-name (catstr "MAKE-" (symbol-name learner-type)))
                   (cons input-dimension learner-params))))
    mulc))

(defun sum-permutation (n m)
  (/ (* (+ n (- n m) 1) m) 2))

(defun index-of-learner (k i L)
  (+ (- k i)
     (sum-permutation (1- L) i)
     -1))

;; TODO: each sub-learner's predict are evaluated twice.
(defun one-vs-one-predict (mulc input)
  (let ((max-cnt 0)
	(max-class nil))
    (loop for k from 0 to (1- (one-vs-one-n-class mulc)) do
      (let ((cnt 0))
	;; negative
	(loop for i from 0 to (1- k) do
          ;; (format t "k: ~A, Negative, learner-index: ~A~%" k (index-of-learner k i (one-vs-one-n-class mulc)))
	  (if (< (funcall (one-vs-one-learner-predict mulc)
                          (svref (one-vs-one-learners-vector mulc)
                                 (index-of-learner k i (one-vs-one-n-class mulc))) input)
		 0d0)
	    (incf cnt)))
	;; positive
	(let ((start-index (sum-permutation (1- (one-vs-one-n-class mulc)) k)))
	  (loop for j from start-index to (+ start-index (- (1- (one-vs-one-n-class mulc)) k 1)) do
            ;; (format t "k: ~A, Positive, learner-index: ~A~%" k j)
	    (if (> (funcall (one-vs-one-learner-predict mulc)
                            (svref (one-vs-one-learners-vector mulc) j) input)
                   0d0)
	      (incf cnt))))
	(if (> cnt max-cnt)
	  (setf max-cnt cnt
		max-class k))))
    max-class))

;; training-label should be integer (0 ... K-1)
(defun one-vs-one-update (mulc input training-label)
  ;; negative
  (loop for i from 0 to (1- training-label) do
    ;; (format t "Negative. Index: ~A~%" (index-of-learner training-label i (one-vs-one-n-class mulc))) ;debug
    (funcall (one-vs-one-learner-update mulc)
             (svref (one-vs-one-learners-vector mulc)
                    (index-of-learner training-label i (one-vs-one-n-class mulc)))
             input -1d0))
  ;; positive
  (let ((start-index (sum-permutation (1- (one-vs-one-n-class mulc)) training-label)))
    (loop for j from start-index to (+ start-index (- (1- (one-vs-one-n-class mulc)) training-label 1)) do
      ;; (format t "Positive. Index: ~A~%" j) ;debug
      (funcall (one-vs-one-learner-update mulc)
               (svref (one-vs-one-learners-vector mulc) j)
               input 1d0))))

(define-multi-class-learner-train/test-functions one-vs-one)
