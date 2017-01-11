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
   :make-lr+sgd :lr+sgd-update :lr+sgd-train :lr+sgd-predict :lr+sgd-test
   :make-lr+adam :lr+adam-update :lr+adam-train :lr+adam-predict :lr+adam-test
   :make-sparse-perceptron :sparse-perceptron-update :sparse-perceptron-train
   :sparse-perceptron-predict :sparse-perceptron-test
   :make-sparse-arow :sparse-arow-update :sparse-arow-train :sparse-arow-predict :sparse-arow-test
   :make-sparse-scw :sparse-scw-update :sparse-scw-train :sparse-scw-predict :sparse-scw-test
   :make-sparse-lr+sgd :sparse-lr+sgd-update :sparse-lr+sgd-train :sparse-lr+sgd-predict :sparse-lr+sgd-test
   :make-sparse-lr+adam :sparse-lr+adam-update :sparse-lr+adam-train :sparse-lr+adam-predict :sparse-lr+adam-test
   :make-one-vs-rest :one-vs-rest-update :one-vs-rest-train :one-vs-rest-predict :one-vs-rest-test
   :make-one-vs-one :one-vs-one-update :one-vs-one-train :one-vs-one-predict :one-vs-one-test))

(in-package :cl-online-learning)

(declaim (type (simple-array double-float 1) *result*))
(defvar *result* (make-dvec 1 0d0))

(defmacro refr ()
  '(aref *result* 0))

(defmacro setr (val)
  `(setf (refr) ,val))

;;; Utils

(defmacro catstr (str1 str2)
  `(concatenate 'string ,str1 ,str2))

;; Signum
(defun sign (x)
  (declare (type double-float x)
           (optimize (speed 3) (safety 0)))
  (if (> x 0d0) 1d0 -1d0))

;; Decision boundary
(defun f (input weight bias)
  (declare (type (simple-array double-float) input weight)
           (type double-float bias)
           (optimize (speed 3) (safety 0)))
  (+ (dot weight input) bias))

(defun f! (input weight bias result)
  (declare (type (simple-array double-float) input weight)
           (type (simple-array double-float 1) result)
           (type double-float bias)
           (optimize (speed 3) (safety 0)))
  (dot! weight input result)
  (setf (aref result 0) (+ (aref result 0) bias))
  (values))

;; Decision boundary (For sparse input)
(defun sf (input weight bias)
  (declare (type clol.vector::sparse-vector input)
           (type (simple-array double-float) weight)
           (type double-float bias)
           (optimize (speed 3) (safety 0)))
  (+ (ds-dot weight input) bias))

(defun sf! (input weight bias result)
  (declare (type clol.vector::sparse-vector input)
           (type (simple-array double-float) weight)
           (type double-float bias)
           (type (simple-array double-float 1) result)
           (optimize (speed 3) (safety 0)))
  (ds-dot! weight input result)
  (setf (aref result 0) (+ (aref result 0) bias))
  (values))

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
       (declare (type ,learner-type ,learner)
                (type ,(if (sparse-symbol? learner-type)
                         'clol.vector::sparse-vector
                         '(simple-array double-float))
                      ,input)
                (type double-float ,training-label)
                (optimize (speed 3) (safety 0)))
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
  (f! input (perceptron-weight learner) (perceptron-bias learner) *result*)
  (when (<= (* training-label (refr)) 0d0)
    (let ((bias (perceptron-bias learner)))
      (declare (type double-float bias))
      (if (> training-label 0d0)
        (progn
          (v+ (perceptron-weight learner) input (perceptron-weight learner))
          (setf (perceptron-bias learner) (+ bias 1d0)))
        (progn
          (v- (perceptron-weight learner) input (perceptron-weight learner))
          (setf (perceptron-bias learner) (- bias 1d0)))))))

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
  (f! input (arow-weight learner) (arow-bias learner) *result*)
  (let ((loss (- 1d0 (* training-label (refr))))
        (sigma0 (arow-sigma0 learner))
        (gamma (arow-gamma learner))
        (bias (arow-bias learner)))
    (declare (type double-float loss sigma0 gamma bias))
    (when (> loss 0d0)
      (dot! (v* (arow-sigma learner) input (arow-tmp-vec1 learner))
            input *result*)
      (let ((beta (/ 1d0 (+ sigma0 (refr) gamma))))
        (declare (type double-float beta))
        (let ((alpha (* loss beta)))
          (declare (type double-float alpha))
          ;; Update weight
          (v*n (arow-tmp-vec1 learner) (* alpha training-label) (arow-tmp-vec2 learner))
          (v+ (arow-weight learner) (arow-tmp-vec2 learner) (arow-weight learner))
          ;; Update bias
          (setf (arow-bias learner) (+ bias (* alpha sigma0 training-label)))
          ;; Update sigma
          (v* (arow-tmp-vec1 learner) (arow-tmp-vec1 learner) (arow-tmp-vec1 learner))
          (v*n (arow-tmp-vec1 learner) beta (arow-tmp-vec1 learner))
          (v- (arow-sigma learner) (arow-tmp-vec1 learner) (arow-sigma learner))
          ;; Update sigma0
          (setf (arow-sigma0 learner)
                (- sigma0 (* beta sigma0 sigma0))))))))

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
  (f! input (scw-weight learner) (scw-bias learner) *result*)
  (let ((m (* training-label (refr)))
        (bias (scw-bias learner))
        (sigma0 (scw-sigma0 learner))
        (phi (scw-phi learner))
        (psi (scw-psi learner))
        (zeta (scw-zeta learner))
        (C (scw-C learner)))
    (declare (type double-float m bias sigma0 phi psi zeta C))
    (dot! (v* (scw-sigma learner) input (scw-tmp-vec1 learner)) input *result*)
    (let ((v (+ sigma0 (refr))))
      (declare (type (double-float 0d0) v))
      (let ((loss (- (* phi (sqrt v)) m)))
        (declare (type double-float loss))
        (when (> loss 0d0)
          (let ((alpha-sqrt-inner (+ (/ (* m m phi phi phi phi) 4d0) (* v phi phi zeta))))
            (declare (type (double-float 0d0) alpha-sqrt-inner))
            (let ((alpha (min C (max 0d0 (- (sqrt alpha-sqrt-inner) (* m psi))))))
              (declare (type double-float alpha))
              (let ((u-sqrt-inner (+ (* alpha alpha v v phi phi) (* 4d0 v))))
                (declare (type (double-float 0d0) u-sqrt-inner))
                (let ((u (let ((base (- (sqrt u-sqrt-inner) (* alpha v phi))))
                           (declare (type double-float base))
                           (/ (* base base) 4d0))))
                  (declare (type (double-float 0d0) u))
                  (let ((beta (/ (* alpha phi) (+ (sqrt u) (* v alpha phi)))))
                    (declare (type double-float beta))
                    ;; Update weight
                    (v*n (scw-tmp-vec1 learner) (* alpha training-label) (scw-tmp-vec2 learner))
                    (v+ (scw-weight learner) (scw-tmp-vec2 learner) (scw-weight learner))
                    ;; Update bias
                    (setf (scw-bias learner) (+ bias (* alpha sigma0 training-label)))
                    ;; Update sigma
                    (v* (scw-tmp-vec1 learner) (scw-tmp-vec1 learner) (scw-tmp-vec1 learner))
                    (v*n (scw-tmp-vec1 learner) beta (scw-tmp-vec1 learner))
                    (v- (scw-sigma learner) (scw-tmp-vec1 learner) (scw-sigma learner))
                    ;; Update sigma0
                    (setf (scw-sigma0 learner)
                          (- sigma0 (* beta sigma0 sigma0)))))))))))))

;;; Logistic regression (L2 regularization)

(defmacro sigmoid (x)
  `(/ 1d0 (+ 1d0 (exp (* -1d0 ,x)))))

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
    (declare (type (double-float 0d0) sigmoid-val))
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

(defun logistic-regression-gradient! (training-label input-vector weight-vector bias C tmp-vec result)
  (declare (type double-float training-label bias C)
           (type (simple-array double-float) input-vector weight-vector tmp-vec result)
           (optimize (speed 3) (safety 0)))
  (f! input-vector weight-vector bias *result*)
  (let ((sigmoid-val (sigmoid (* training-label (refr)))))
    (declare (type (double-float 0d0) sigmoid-val))
    ;; set gradient-vector to result
    (v*n input-vector
         (* (- 1d0 sigmoid-val) (* -1d0 training-label))
         tmp-vec)
    (v*n weight-vector (* 2d0 C) result)
    (v+ tmp-vec result result)
    ;; return g0
    (setr (+ (* (- 1d0 sigmoid-val)
                (* -1d0 training-label))
             (* 2d0 C bias)))
    (values)))

(defstruct (lr+sgd (:constructor %make-lr+sgd))
  input-dimension weight bias
  ;; meta parameters
  C eta g tmp-vec)

(defun make-lr+sgd (input-dimension C eta)
  (%make-lr+sgd
   :input-dimension input-dimension
   :weight (make-dvec input-dimension 0d0)
   :bias 0d0
   :C C
   :eta eta
   :g (make-dvec input-dimension 0d0)
   :tmp-vec (make-dvec input-dimension 0d0)))

(define-learner lr+sgd (learner input training-label)
  (let ((weight (lr+sgd-weight learner))
        (bias (lr+sgd-bias learner))
        (C (lr+sgd-C learner))
        (eta (lr+sgd-eta learner))
        (tmp-vec (lr+sgd-tmp-vec learner))
        (g (lr+sgd-g learner)))
    (declare (type double-float bias C eta)
             (type (simple-array double-float) weight tmp-vec g))
    ;; calc g (gradient)
    (logistic-regression-gradient! training-label input weight bias C tmp-vec g)
    (v*n g eta g)
    (v- weight g weight)
    (setf (lr+sgd-bias learner) (- bias (* eta (refr))))))

;; Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980)
(defstruct (lr+adam (:constructor %make-lr+adam)
                 (:print-object %print-lr+adam))
  input-dimension weight bias
  ;; meta parameters
  C alpha epsilon beta1 beta2
  ;; internal parameters
  g m v m0 v0 beta1^t beta2^t tmp-vec)

(defun %print-lr+adam (obj stream)
  (format stream "#S(LR+ADAM~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A)"
          (lr+adam-input-dimension obj)
          (let ((w (lr+adam-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (lr+adam-bias obj)))

(defun make-lr+adam (input-dimension C alpha epsilon beta1 beta2)
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
  (%make-lr+adam
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

(define-learner lr+adam (learner input training-label)
  (let ((weight (lr+adam-weight learner)) (bias (lr+adam-bias learner))
        (C (lr+adam-C learner)) (tmp-vec (lr+adam-tmp-vec learner))
        (g (lr+adam-g learner)) (g0 0d0)
        (m (lr+adam-m learner)) (m0 (lr+adam-m0 learner))
        (v (lr+adam-v learner)) (v0 (lr+adam-v0 learner))
        (alpha (lr+adam-alpha learner))
        (beta1 (lr+adam-beta1 learner)) (beta2 (lr+adam-beta2 learner))
        (beta1^t (lr+adam-beta1^t learner)) (beta2^t (lr+adam-beta2^t learner))
        (epsilon (lr+adam-epsilon learner)))
    (declare (type double-float bias C g0 m0 v0 alpha beta1 beta2 beta1^t beta2^t epsilon)
             (type (simple-array double-float) weight tmp-vec g m v)
             (optimize (speed 3) (safety 0)))
    ;; calc g (gradient)
    (logistic-regression-gradient! training-label input weight bias C tmp-vec g)
    (setf g0 (refr))
    ;; update m_t from m_t-1
    (v*n m beta1 m)
    (v*n g (- 1d0 beta1) tmp-vec)
    (v+ m tmp-vec m)
    ;; calc g^2 (gradient^2)
    (v* g g g)
    ;; update v_t from v_t-1
    (v*n v beta2 v)
    (v*n g (- 1d0 beta2) tmp-vec)
    (v+ v tmp-vec v)
    ;; update m0 and v0
    (let ((new-m0 (+ (* beta1 m0) (* (- 1d0 beta1) g0)))
          (new-v0 (+ (* beta2 v0) (* (- 1d0 beta2) (* g0 g0))))
          (epsilon-coefficient-sqrt-inner (- 1d0 beta2^t)))
      (declare (type double-float new-m0)
               (type (double-float 0d0) new-v0 epsilon-coefficient-sqrt-inner))
      ;; update weight
      (let* ((epsilon-coefficient (sqrt epsilon-coefficient-sqrt-inner))
             (epsilon^ (* epsilon-coefficient epsilon))
             (alpha_t (* alpha (/ epsilon-coefficient (- 1d0 beta1^t)))))
        (v-sqrt v tmp-vec)
        (v+n tmp-vec epsilon^ tmp-vec)
        (v/ m tmp-vec tmp-vec)
        (v*n tmp-vec alpha_t tmp-vec)
        (v- weight tmp-vec weight)
        ;; update m0, v0, and bias
        (setf (lr+adam-m0 learner) new-m0
              (lr+adam-v0 learner) new-v0
              (lr+adam-bias learner) (- bias (* alpha_t (/ new-m0 (+ (sqrt new-v0) epsilon^)))))))
    ;; update beta1^2 and beta2^2
    (setf (lr+adam-beta1^t learner) (* beta1 beta1^t)
          (lr+adam-beta2^t learner) (* beta2 beta2^t))))

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
  (sf! input (sparse-perceptron-weight learner) (sparse-perceptron-bias learner) *result*)
  (when (<= (* training-label (refr)) 0d0)
    (let ((bias (sparse-perceptron-bias learner)))
      (declare (type double-float bias))
      (if (> training-label 0d0)
        (progn
          (ds-v+ (sparse-perceptron-weight learner) input (sparse-perceptron-weight learner))
          (setf (sparse-perceptron-bias learner) (+ bias 1d0)))
        (progn
          (ds-v- (sparse-perceptron-weight learner) input (sparse-perceptron-weight learner))
          (setf (sparse-perceptron-bias learner) (- bias 1d0)))))))

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
  (sf! input (sparse-arow-weight learner) (sparse-arow-bias learner) *result*)
  (let ((index-vector (sparse-vector-index-vector input))
        (loss (- 1d0 (* training-label (refr))))
        (bias (sparse-arow-bias learner))
        (sigma0 (sparse-arow-sigma0 learner))
        (gamma (sparse-arow-gamma learner)))
    (declare (type (simple-array fixnum) index-vector)
             (type double-float loss bias sigma0 gamma))
    (when (> loss 0d0)
      (ds-dot! (ds-v* (sparse-arow-sigma learner) input (sparse-arow-tmp-vec1 learner))
               input *result*)
      (let ((beta (/ 1d0 (+ sigma0 (refr) gamma))))
        (declare (type double-float beta))
        (let ((alpha (* loss beta)))
          (declare (type double-float alpha))
          ;; Update weight
          (ps-v*n (sparse-arow-tmp-vec1 learner) (* alpha training-label) index-vector
                  (sparse-arow-tmp-vec2 learner))
          (dps-v+ (sparse-arow-weight learner) (sparse-arow-tmp-vec2 learner) index-vector
                  (sparse-arow-weight learner))
          ;; Update bias
          (setf (sparse-arow-bias learner) (+ bias (* alpha sigma0 training-label)))
          ;; Update sigma
          (dps-v* (sparse-arow-tmp-vec1 learner) (sparse-arow-tmp-vec1 learner) index-vector
                  (sparse-arow-tmp-vec1 learner))
          (ps-v*n (sparse-arow-tmp-vec1 learner) beta index-vector
                  (sparse-arow-tmp-vec1 learner))
          (dps-v- (sparse-arow-sigma learner) (sparse-arow-tmp-vec1 learner) index-vector
                  (sparse-arow-sigma learner))
          ;; Update sigma0
          (setf (sparse-arow-sigma0 learner)
                (- sigma0 (* beta sigma0 sigma0))))))))

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
  (sf! input (sparse-scw-weight learner) (sparse-scw-bias learner) *result*)
  (let ((index-vector (sparse-vector-index-vector input))
        (m (* training-label (refr)))
        (bias (sparse-scw-bias learner))
        (sigma0 (sparse-scw-sigma0 learner))
        (phi (sparse-scw-phi learner))
        (psi (sparse-scw-psi learner))
        (zeta (sparse-scw-zeta learner))
        (C (sparse-scw-C learner)))
    (declare (type (simple-array fixnum) index-vector)
             (type double-float m bias sigma0 phi psi zeta C))
    (ds-dot! (ds-v* (sparse-scw-sigma learner) input (sparse-scw-tmp-vec1 learner)) input *result*)
    (let ((v (+ sigma0 (refr))))
      (declare (type (double-float 0d0) v))
      (let ((loss (- (* phi (sqrt v)) m)))
        (declare (type double-float loss))
        (when (> loss 0d0)
          (let ((alpha-sqrt-inner (+ (/ (* m m phi phi phi phi) 4d0) (* v phi phi zeta))))
            (declare (type (double-float 0d0) alpha-sqrt-inner))
            (let ((alpha (min C (max 0d0 (- (sqrt alpha-sqrt-inner) (* m psi))))))
              (declare (type double-float alpha))
              (let ((u-sqrt-inner (+ (* alpha alpha v v phi phi) (* 4d0 v))))
                (declare (type (double-float 0d0) u-sqrt-inner))
                (let ((u (let ((base (- (sqrt u-sqrt-inner) (* alpha v phi))))
                           (declare (type double-float base))
                           (/ (* base base) 4d0))))
                  (declare (type (double-float 0d0) u))
                  (let ((beta (/ (* alpha phi) (+ (sqrt u) (* v alpha phi)))))
                    (declare (type double-float beta))
                    ;; Update weight
                    (ps-v*n (sparse-scw-tmp-vec1 learner) (* alpha training-label) index-vector
                            (sparse-scw-tmp-vec2 learner))
                    (dps-v+ (sparse-scw-weight learner) (sparse-scw-tmp-vec2 learner) index-vector
                            (sparse-scw-weight learner))
                    ;; Update bias
                    (setf (sparse-scw-bias learner) (+ bias (* alpha sigma0 training-label)))
                    ;; Update sigma
                    (dps-v* (sparse-scw-tmp-vec1 learner) (sparse-scw-tmp-vec1 learner) index-vector
                            (sparse-scw-tmp-vec1 learner))
                    (ps-v*n (sparse-scw-tmp-vec1 learner) beta index-vector
                            (sparse-scw-tmp-vec1 learner))
                    (dps-v- (sparse-scw-sigma learner) (sparse-scw-tmp-vec1 learner) index-vector
                            (sparse-scw-sigma learner))
                    ;; Update sigma0
                    (setf (sparse-scw-sigma0 learner)
                          (- sigma0 (* beta sigma0 sigma0)))))))))))))

;;; Logistic regression model (Sparse)

;; tmp-vec is pseudosparse-vector,

(defun logistic-regression-gradient-sparse!
    (training-label input-vector weight-vector bias C tmp-vec result)
  (declare (type double-float training-label bias C)
           (type clol.vector::sparse-vector input-vector)
           (type (simple-array double-float) weight-vector tmp-vec result)
           (optimize (speed 3) (safety 0)))
  (sf! input-vector weight-vector bias *result*)
  (let ((sigmoid-val (sigmoid (* training-label (refr)))))
    (declare (type (double-float 0d0) sigmoid-val))
    ;; set gradient-vector to result
    (sps-v*n input-vector
             (* (- 1d0 sigmoid-val) (* -1d0 training-label))
             tmp-vec)
    (v*n weight-vector (* 2d0 C) result)
    (dps-v+ result tmp-vec (sparse-vector-index-vector input-vector) result)
    ;; return g0
    (setr (+ (* (- 1d0 sigmoid-val)
                (* -1d0 training-label))
             (* 2d0 C bias)))
    (values)))

;;; Sparse lr+sgd

(defstruct (sparse-lr+sgd (:constructor %make-sparse-lr+sgd))
  input-dimension weight bias
  ;; meta parameters
  C eta g tmp-vec)

(defun make-sparse-lr+sgd (input-dimension C eta)
  (%make-sparse-lr+sgd
   :input-dimension input-dimension
   :weight (make-dvec input-dimension 0d0)
   :bias 0d0
   :C C
   :eta eta
   :g (make-dvec input-dimension 0d0)
   :tmp-vec (make-dvec input-dimension 0d0)))

(define-learner sparse-lr+sgd (learner input training-label)
  (let ((weight (sparse-lr+sgd-weight learner))
        (bias (sparse-lr+sgd-bias learner))
        (C (sparse-lr+sgd-C learner))
        (eta (sparse-lr+sgd-eta learner))
        (tmp-vec (sparse-lr+sgd-tmp-vec learner))
        (g (sparse-lr+sgd-g learner)))
    (declare (type double-float bias C eta)
             (type (simple-array double-float) weight tmp-vec g))
    ;; calc g (gradient)
    (logistic-regression-gradient-sparse! training-label input weight bias C tmp-vec g)
    (v*n g eta g)
    (v- weight g weight)
    (setf (sparse-lr+sgd-bias learner) (- bias (* eta (refr))))))

;;; Sparse lr+adam

(defstruct (sparse-lr+adam (:constructor %make-sparse-lr+adam)
                           (:print-object %print-sparse-lr+adam))
  input-dimension weight bias
  ;; meta parameters
  C alpha epsilon beta1 beta2
  ;; internal parameters
  g m v m0 v0 beta1^t beta2^t tmp-vec)

(defun %print-sparse-lr+adam (obj stream)
  (format stream "#S(SPARSE-LR+ADAM~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A)"
          (sparse-lr+adam-input-dimension obj)
          (let ((w (sparse-lr+adam-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (sparse-lr+adam-bias obj)))

(defun make-sparse-lr+adam (input-dimension C alpha epsilon beta1 beta2)
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
  (%make-sparse-lr+adam
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

(define-learner sparse-lr+adam (learner input training-label)
  (let ((weight (sparse-lr+adam-weight learner)) (bias (sparse-lr+adam-bias learner))
        (C (sparse-lr+adam-C learner)) (tmp-vec (sparse-lr+adam-tmp-vec learner))
        (g (sparse-lr+adam-g learner)) (g0 0d0)
        (m (sparse-lr+adam-m learner)) (m0 (sparse-lr+adam-m0 learner))
        (v (sparse-lr+adam-v learner)) (v0 (sparse-lr+adam-v0 learner))
        (alpha (sparse-lr+adam-alpha learner))
        (beta1 (sparse-lr+adam-beta1 learner)) (beta2 (sparse-lr+adam-beta2 learner))
        (beta1^t (sparse-lr+adam-beta1^t learner)) (beta2^t (sparse-lr+adam-beta2^t learner))
        (epsilon (sparse-lr+adam-epsilon learner)))
    (declare (type double-float bias C g0 m0 v0 alpha beta1 beta2 beta1^t beta2^t epsilon)
             (type (simple-array double-float) weight tmp-vec g m v)
             (optimize (speed 3) (safety 0)))
    ;; calc g (gradient)
    (logistic-regression-gradient-sparse! training-label input weight bias C tmp-vec g)
    (setf g0 (refr))
    ;; update m_t from m_t-1
    (v*n m beta1 m)
    (v*n g (- 1d0 beta1) tmp-vec)
    (v+ m tmp-vec m)
    ;; calc g^2 (gradient^2)
    (v* g g g)
    ;; update v_t from v_t-1
    (v*n v beta2 v)
    (v*n g (- 1d0 beta2) tmp-vec)
    (v+ v tmp-vec v)
    ;; update m0 and v0
    (let ((new-m0 (+ (* beta1 m0) (* (- 1d0 beta1) g0)))
          (new-v0 (+ (* beta2 v0) (* (- 1d0 beta2) (* g0 g0))))
          (epsilon-coefficient-sqrt-inner (- 1d0 beta2^t)))
      (declare (type double-float new-m0)
               (type (double-float 0d0) new-v0 epsilon-coefficient-sqrt-inner))
      ;; update weight
      (let* ((epsilon-coefficient (sqrt epsilon-coefficient-sqrt-inner))
             (epsilon^ (* epsilon-coefficient epsilon))
             (alpha_t (* alpha (/ epsilon-coefficient (- 1d0 beta1^t)))))
        (v-sqrt v tmp-vec)
        (v+n tmp-vec epsilon^ tmp-vec)
        (v/ m tmp-vec tmp-vec)
        (v*n tmp-vec alpha_t tmp-vec)
        (v- weight tmp-vec weight)
        ;; update m0, v0, and bias
        (setf (sparse-lr+adam-m0 learner) new-m0
              (sparse-lr+adam-v0 learner) new-v0
              (sparse-lr+adam-bias learner) (- bias (* alpha_t (/ new-m0 (+ (sqrt new-v0) epsilon^)))))))
    ;; update beta1^2 and beta2^2
    (setf (sparse-lr+adam-beta1^t learner) (* beta1 beta1^t)
          (sparse-lr+adam-beta2^t learner) (* beta2 beta2^t))))

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
