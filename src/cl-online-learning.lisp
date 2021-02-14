;;; -*- coding:utf-8; mode:lisp -*-

;;; search difference of efficiency struct and CLOS

(in-package :cl-user)
(defpackage :cl-online-learning
  (:use :cl :cl-online-learning.vector)
  (:nicknames :clol)
  (:export
   :train :test :dim-of :n-class-of :sparse-learner?
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
   :make-one-vs-one :one-vs-one-update :one-vs-one-train :one-vs-one-predict :one-vs-one-test
   ;; regression
   :make-rls :rls-update :rls-train :rls-predict :rls-test
   :make-sparse-rls :sparse-rls-update :sparse-rls-train :sparse-rls-predict :sparse-rls-test
   ; save/restore
   :save :restore))

(in-package :cl-online-learning)

;;; Utils

(defmacro catstr (str1 str2)
  `(concatenate 'string ,str1 ,str2))

;; Signum
(defun sign (x)
  (declare (type single-float x)
           (optimize (speed 3) (safety 0)))
  (if (> x 0.0) 1.0 -1.0))

;; Decision boundary
(defun f (input weight bias)
  (declare (type (simple-array single-float) input weight)
           (type single-float bias)
           (optimize (speed 3) (safety 0)))
  (+ (dot weight input) bias))

(defun f! (input weight bias result)
  (declare (type (simple-array single-float) input weight)
           (type (simple-array single-float 1) result)
           (type single-float bias)
           (optimize (speed 3) (safety 0)))
  (dot! weight input result)
  (setf (aref result 0) (+ (aref result 0) bias))
  (values))

;; Decision boundary (For sparse input)
(defun sf (input weight bias)
  (declare (type sparse-vector input)
           (type (simple-array single-float) weight)
           (type single-float bias)
           (optimize (speed 3) (safety 0)))
  (+ (ds-dot weight input) bias))

(defun sf! (input weight bias result)
  (declare (type sparse-vector input)
           (type (simple-array single-float) weight)
           (type (simple-array single-float 1) result)
           (type single-float bias)
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
                         'sparse-vector
                         '(simple-array single-float))
                      ,input)
                (type single-float ,training-label)
                (optimize (speed 3) (safety 0))
                )
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
         (learner test-data &key (quiet-p nil) (stream nil))
       (let* ((len (length test-data))
              (n-correct (count-if
                          (lambda (datum)
                            (let ((predict (,(intern (catstr (symbol-name learner-type) "-PREDICT"))
                                            learner (cdr datum))))
                              (format stream "~D~%" (round predict))
                              (= predict (car datum))))
                          test-data))
              (accuracy (* (/ n-correct len) 100.0)))
         (if (not quiet-p)
           (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" accuracy n-correct len))
         (values accuracy n-correct len)))))

(defun train (learner training-data)
  (funcall (intern (catstr (symbol-name (type-of learner)) "-TRAIN")
                   :cl-online-learning)
           learner training-data))

(defun test (learner test-data &key (quiet-p nil) (stream nil))
  (funcall (intern (catstr (symbol-name (type-of learner)) "-TEST")
                   :cl-online-learning)
           learner test-data :quiet-p quiet-p :stream stream))

(defun dim-of (learner)
  (let ((learner
          (typecase learner
            (one-vs-one  (aref (one-vs-one-learners-vector learner) 0))
            (one-vs-rest (aref (one-vs-rest-learners-vector learner) 0))
            (t learner))))
    (length (funcall (intern (catstr (symbol-name (type-of learner)) "-WEIGHT")
                             :cl-online-learning)
                     learner))))

(defun n-class-of (learner)
  (typecase learner
    (one-vs-one  (one-vs-one-n-class learner))
    (one-vs-rest (one-vs-rest-n-class learner))
    (t 2)))

(defun sparse-learner? (learner)
  (typecase learner
    (one-vs-one  (sparse-symbol? (type-of (aref (one-vs-one-learners-vector learner) 0))))
    (one-vs-rest (sparse-symbol? (type-of (aref (one-vs-rest-learners-vector learner) 0))))
    (t (sparse-symbol? (type-of learner)))))

;;; Perceptron

(defstruct (perceptron (:constructor %make-perceptron)
                       (:print-object %print-perceptron))
  input-dimension weight bias tmp-float)

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
                    :weight (make-vec input-dimension 0.0)
                    :bias 0.0
                    :tmp-float (make-vec 1 0.0)))

(define-learner perceptron (learner input training-label)
  (let ((tmp-float (perceptron-tmp-float learner)))
    (declare (type (simple-array single-float 1) tmp-float))
    (f! input (perceptron-weight learner) (perceptron-bias learner) tmp-float)
    (when (<= (* training-label (aref tmp-float 0)) 0.0)
      (let ((bias (perceptron-bias learner)))
        (declare (type single-float bias))
        (if (> training-label 0.0)
          (progn
            (v+ (perceptron-weight learner) input (perceptron-weight learner))
            (setf (perceptron-bias learner) (+ bias 1.0)))
          (progn
            (v- (perceptron-weight learner) input (perceptron-weight learner))
            (setf (perceptron-bias learner) (- bias 1.0))))))))

;;; AROW

(defstruct (arow (:constructor  %make-arow)
                 (:print-object %print-arow))
  input-dimension weight bias
  gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-float)

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
  (check-type gamma number)
  (%make-arow :input-dimension input-dimension
              :weight (make-vec input-dimension 0.0) ; mu
              :bias 0.0                              ; mu0
              :gamma (coerce gamma 'single-float)
              :sigma (make-vec input-dimension 1.0)
              :sigma0 1.0
              :tmp-vec1 (make-vec input-dimension 0.0)
              :tmp-vec2 (make-vec input-dimension 0.0)
              :tmp-float (make-vec 1 0.0)))

(define-learner arow (learner input training-label)
  (let ((tmp-float (arow-tmp-float learner)))
    (declare (type (simple-array single-float 1) tmp-float))
    (f! input (arow-weight learner) (arow-bias learner) tmp-float)
    (let ((loss (- 1.0 (* training-label (aref tmp-float 0))))
          (sigma0 (arow-sigma0 learner))
          (gamma (arow-gamma learner))
          (bias (arow-bias learner)))
      (declare (type single-float loss sigma0 gamma bias))
      (when (> loss 0.0)
        (dot! (v* (arow-sigma learner) input (arow-tmp-vec1 learner))
              input tmp-float)
        (let ((beta (/ 1.0 (+ sigma0 (aref tmp-float 0) gamma))))
          (declare (type single-float beta))
          (let ((alpha (* loss beta)))
            (declare (type single-float alpha))
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
                  (- sigma0 (* beta sigma0 sigma0)))))))))

;;; SCW-I

;; Approximation of error function
(defun inverse-erf (x)
  (let* ((a (/ (* 8.0 (- pi 3.0))
	       (* 3.0 pi (- 4.0 pi))))
	 (c2/pia (/ 2.0 pi a))
	 (ln1-x^2 (log (- 1.0 (* x x))))
	 (comp (+ c2/pia (/ ln1-x^2 2.0))))
    (* (sign x)
       (sqrt (- (sqrt (- (* comp comp) (/ ln1-x^2 a)))
                comp)))))

(defun probit (p)
  (* (sqrt 2.0)
     (inverse-erf (- (* 2.0 p) 1.0))))

(defstruct (scw (:constructor  %make-scw)
                (:print-object %print-scw))
  input-dimension weight bias
  eta C
  ;; Internal parameters
  phi psi zeta sigma sigma0
  tmp-vec1 tmp-vec2 tmp-float)

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
  (check-type eta number)
  (check-type C number)
  (assert (< 0.0 eta 1.0))
  (let* ((eta (coerce eta 'single-float))
         (C (coerce C 'single-float))
         (phi (coerce (probit eta) 'single-float))
	 (psi (+ 1.0 (/ (* phi phi) 2.0)))
	 (zeta (+ 1.0 (* phi phi))))
    (%make-scw
     :input-dimension input-dimension
     :weight (make-vec input-dimension 0.0)
     :bias 0.0  :eta eta  :C C
     :phi phi   :psi psi  :zeta zeta
     :sigma    (make-vec input-dimension 1.0)
     :sigma0 1.0
     :tmp-vec1 (make-vec input-dimension 0.0)
     :tmp-vec2 (make-vec input-dimension 0.0)
     :tmp-float (make-vec 1 0.0))))

(define-learner scw (learner input training-label)
  (let ((tmp-float (scw-tmp-float learner)))
    (declare (type (simple-array single-float 1) tmp-float))
    (f! input (scw-weight learner) (scw-bias learner) tmp-float)
    (let ((m (* training-label (aref tmp-float 0)))
          (bias (scw-bias learner))
          (sigma0 (scw-sigma0 learner))
          (phi (scw-phi learner))
          (psi (scw-psi learner))
          (zeta (scw-zeta learner))
          (C (scw-C learner)))
      (declare (type single-float m bias sigma0 phi psi zeta C))
      (dot! (v* (scw-sigma learner) input (scw-tmp-vec1 learner)) input tmp-float)
      (let ((v (+ sigma0 (aref tmp-float 0))))
        (declare (type (single-float 0.0) v))
        (let ((loss (- (* phi (sqrt v)) m)))
          (declare (type single-float loss))
          (when (> loss 0.0)
            (let ((alpha-sqrt-inner (+ (/ (* m m phi phi phi phi) 4.0) (* v phi phi zeta))))
              (declare (type (single-float 0.0) alpha-sqrt-inner))
              (let ((alpha (min C (max 0.0 (- (sqrt alpha-sqrt-inner) (* m psi))))))
                (declare (type single-float alpha))
                (let ((u-sqrt-inner (+ (* alpha alpha v v phi phi) (* 4.0 v))))
                  (declare (type (single-float 0.0) u-sqrt-inner))
                  (let ((u (let ((base (- (sqrt u-sqrt-inner) (* alpha v phi))))
                             (declare (type single-float base))
                             (/ (* base base) 4.0))))
                    (declare (type (single-float 0.0) u))
                    (let ((beta (/ (* alpha phi) (+ (sqrt u) (* v alpha phi)))))
                      (declare (type single-float beta))
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
                            (- sigma0 (* beta sigma0 sigma0))))))))))))))

;;; Logistic regression (L2 regularization)

(defmacro sigmoid (x)
  `(/ 1.0 (+ 1.0 (exp (* -1.0 ,x)))))

(defun logistic-regression-gradient! (training-label input-vector weight-vector bias C tmp-vec g-result g0-result)
  (declare (type single-float training-label bias C)
           (type (simple-array single-float) input-vector weight-vector tmp-vec g-result)
           (type (simple-array single-float 1) g0-result)
           (optimize (speed 3) (safety 0)))
  (f! input-vector weight-vector bias g0-result)
  (let ((sigmoid-val (sigmoid (* training-label (aref g0-result 0)))))
    (declare (type (single-float 0.0) sigmoid-val))
    ;; set gradient-vector to g-result
    (v*n input-vector
         (* (- 1.0 sigmoid-val) (* -1.0 training-label))
         tmp-vec)
    (v*n weight-vector (* 2.0 C) g-result)
    (v+ tmp-vec g-result g-result)
    ;; return g0
    (setf (aref g0-result 0)
          (+ (* (- 1.0 sigmoid-val)
                (* -1.0 training-label))
             (* 2.0 C bias)))
    (values)))

(defstruct (lr+sgd (:constructor %make-lr+sgd))
  input-dimension weight bias
  ;; meta parameters
  C eta g tmp-vec tmp-float)

(defun make-lr+sgd (input-dimension C eta)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type C number)
  (check-type eta number)
  (let* ((C (coerce C 'single-float))
         (eta (coerce eta 'single-float)))
    (%make-lr+sgd
     :input-dimension input-dimension
     :weight (make-vec input-dimension 0.0)
     :bias 0.0
     :C C
     :eta eta
     :g (make-vec input-dimension 0.0)
     :tmp-vec (make-vec input-dimension 0.0)
     :tmp-float (make-vec 1 0.0))))

(define-learner lr+sgd (learner input training-label)
  (let ((weight (lr+sgd-weight learner))
        (bias (lr+sgd-bias learner))
        (C (lr+sgd-C learner))
        (eta (lr+sgd-eta learner))
        (tmp-vec (lr+sgd-tmp-vec learner))
        (g (lr+sgd-g learner))
        (tmp-float (lr+sgd-tmp-float learner)))
    (declare (type single-float bias C eta)
             (type (simple-array single-float) weight tmp-vec g)
             (type (simple-array single-float 1) tmp-float))
    ;; calc g (gradient)
    (logistic-regression-gradient! training-label input weight bias C tmp-vec g tmp-float)
    (v*n g eta g)
    (v- weight g weight)
    (setf (lr+sgd-bias learner) (- bias (* eta (aref tmp-float 0))))))

;; Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980)
(defstruct (lr+adam (:constructor %make-lr+adam)
                 (:print-object %print-lr+adam))
  input-dimension weight bias
  ;; meta parameters
  C alpha epsilon beta1 beta2
  ;; internal parameters
  g m v m0 v0 beta1^t beta2^t tmp-vec tmp-float)

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
  (check-type C number)
  (check-type alpha number)
  (check-type epsilon number)
  (check-type beta1 number)
  (check-type beta2 number)
  (assert (< 0.0 alpha))
  (assert (and (<= 0.0 beta1) (< beta1 1.0)))
  (assert (and (<= 0.0 beta2) (< beta2 1.0)))
  (%make-lr+adam
   :input-dimension input-dimension
   :weight (make-vec input-dimension 0.0)
   :bias 0.0
   :C (coerce C 'single-float)
   :alpha (coerce alpha 'single-float)
   :epsilon (coerce epsilon 'single-float)
   :beta1 (coerce beta1 'single-float)
   :beta2 (coerce beta2 'single-float)
   :g (make-vec input-dimension 0.0)
   :m (make-vec input-dimension 0.0)
   :v (make-vec input-dimension 0.0)
   :m0 0.0
   :v0 0.0
   :beta1^t beta1
   :beta2^t beta2
   :tmp-vec (make-vec input-dimension 0.0)
   :tmp-float (make-vec 1 0.0)))

(define-learner lr+adam (learner input training-label)
  (let ((weight (lr+adam-weight learner)) (bias (lr+adam-bias learner))
        (C (lr+adam-C learner)) (tmp-vec (lr+adam-tmp-vec learner)) (tmp-float (lr+adam-tmp-float learner))
        (g (lr+adam-g learner)) (g0 0.0)
        (m (lr+adam-m learner)) (m0 (lr+adam-m0 learner))
        (v (lr+adam-v learner)) (v0 (lr+adam-v0 learner))
        (alpha (lr+adam-alpha learner))
        (beta1 (lr+adam-beta1 learner)) (beta2 (lr+adam-beta2 learner))
        (beta1^t (lr+adam-beta1^t learner)) (beta2^t (lr+adam-beta2^t learner))
        (epsilon (lr+adam-epsilon learner)))
    (declare (type single-float bias C g0 m0 v0 alpha beta1 beta2 beta1^t beta2^t epsilon)
             (type (simple-array single-float) weight tmp-vec g m v)
             (type (simple-array single-float 1) tmp-float)
             (optimize (speed 3) (safety 0)))
    ;; calc g (gradient)
    (logistic-regression-gradient! training-label input weight bias C tmp-vec g tmp-float)
    (setf g0 (aref tmp-float 0))
    ;; update m_t from m_t-1
    (v*n m beta1 m)
    (v*n g (- 1.0 beta1) tmp-vec)
    (v+ m tmp-vec m)
    ;; calc g^2 (gradient^2)
    (v* g g g)
    ;; update v_t from v_t-1
    (v*n v beta2 v)
    (v*n g (- 1.0 beta2) tmp-vec)
    (v+ v tmp-vec v)
    ;; update m0 and v0
    (let ((new-m0 (+ (* beta1 m0) (* (- 1.0 beta1) g0)))
          (new-v0 (+ (* beta2 v0) (* (- 1.0 beta2) (* g0 g0))))
          (epsilon-coefficient-sqrt-inner (- 1.0 beta2^t)))
      (declare (type single-float new-m0)
               (type (single-float 0.0) new-v0 epsilon-coefficient-sqrt-inner))
      ;; update weight
      (let* ((epsilon-coefficient (sqrt epsilon-coefficient-sqrt-inner))
             (epsilon^ (* epsilon-coefficient epsilon))
             (alpha_t (* alpha (/ epsilon-coefficient (- 1.0 beta1^t)))))
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
  input-dimension weight bias tmp-float)

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
                           :weight (make-vec input-dimension 0.0)
                           :bias 0.0
                           :tmp-float (make-vec 1 0.0)))

(define-learner sparse-perceptron (learner input training-label)
  (let ((tmp-float (sparse-perceptron-tmp-float learner)))
    (declare (type (simple-array single-float 1) tmp-float))
    (sf! input (sparse-perceptron-weight learner) (sparse-perceptron-bias learner) tmp-float)
    (when (<= (* training-label (aref tmp-float 0)) 0.0)
      (let ((bias (sparse-perceptron-bias learner)))
        (declare (type single-float bias))
        (if (> training-label 0.0)
          (progn
            (ds-v+ (sparse-perceptron-weight learner) input (sparse-perceptron-weight learner))
            (setf (sparse-perceptron-bias learner) (+ bias 1.0)))
          (progn
            (ds-v- (sparse-perceptron-weight learner) input (sparse-perceptron-weight learner))
            (setf (sparse-perceptron-bias learner) (- bias 1.0))))))))

;;; Sparse AROW

(defstruct (sparse-arow (:constructor  %make-sparse-arow)
                        (:print-object %print-sparse-arow))
  input-dimension weight bias
  gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-float)

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
  (check-type gamma number)
  (%make-sparse-arow :input-dimension input-dimension
                     :weight (make-vec input-dimension 0.0) ; mu
                     :bias 0.0                               ; mu0
                     :gamma (coerce gamma 'single-float)
                     :sigma (make-vec input-dimension 1.0)
                     :sigma0 1.0
                     :tmp-vec1 (make-vec input-dimension 0.0)
                     :tmp-vec2 (make-vec input-dimension 0.0)
                     :tmp-float (make-vec 1 0.0)))

(define-learner sparse-arow (learner input training-label)
  (let ((tmp-float (sparse-arow-tmp-float learner)))
    (declare (type (simple-array single-float 1) tmp-float))
    (sf! input (sparse-arow-weight learner) (sparse-arow-bias learner) tmp-float)
    (let ((index-vector (sparse-vector-index-vector input))
          (loss (- 1.0 (* training-label (aref tmp-float 0))))
          (bias (sparse-arow-bias learner))
          (sigma0 (sparse-arow-sigma0 learner))
          (gamma (sparse-arow-gamma learner)))
      (declare (type (simple-array fixnum) index-vector)
               (type single-float loss bias sigma0 gamma))
      (when (> loss 0.0)
        (ds-dot! (ds-v* (sparse-arow-sigma learner) input (sparse-arow-tmp-vec1 learner)) input tmp-float)
        (let ((beta (/ 1.0 (+ sigma0 (aref tmp-float 0) gamma))))
          (declare (type single-float beta))
          (let ((alpha (* loss beta)))
            (declare (type single-float alpha))
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
                  (- sigma0 (* beta sigma0 sigma0)))))))))

;;; Sparse SCW-I

(defstruct (sparse-scw (:constructor  %make-sparse-scw)
                       (:print-object %print-sparse-scw))
  input-dimension weight bias
  eta C
  ;; Internal parameters
  phi psi zeta sigma sigma0
  tmp-vec1 tmp-vec2 tmp-float)

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
  (check-type eta number)
  (check-type C number)
  (assert (< 0.0 eta 1.0))
  (let* ((eta (coerce eta 'single-float))
         (C (coerce C 'single-float))
         (phi (coerce (probit eta) 'single-float))
	 (psi (+ 1.0 (/ (* phi phi) 2.0)))
	 (zeta (+ 1.0 (* phi phi))))
    (%make-sparse-scw
     :input-dimension input-dimension
     :weight (make-vec input-dimension 0.0)
     :bias 0.0
     :eta eta
     :C C
     :phi phi
     :psi psi
     :zeta zeta
     :sigma (make-vec input-dimension 1.0)
     :sigma0 1.0
     :tmp-vec1 (make-vec input-dimension 0.0)
     :tmp-vec2 (make-vec input-dimension 0.0)
     :tmp-float (make-vec 1 0.0))))

(define-learner sparse-scw (learner input training-label)
  (let ((tmp-float (sparse-scw-tmp-float learner)))
    (declare (type (simple-array single-float 1) tmp-float))
    (sf! input (sparse-scw-weight learner) (sparse-scw-bias learner) tmp-float)
    (let ((index-vector (sparse-vector-index-vector input))
          (m (* training-label (aref tmp-float 0)))
          (bias (sparse-scw-bias learner))
          (sigma0 (sparse-scw-sigma0 learner))
          (phi (sparse-scw-phi learner))
          (psi (sparse-scw-psi learner))
          (zeta (sparse-scw-zeta learner))
          (C (sparse-scw-C learner)))
      (declare (type (simple-array fixnum) index-vector)
               (type single-float m bias sigma0 phi psi zeta C))
      (ds-dot! (ds-v* (sparse-scw-sigma learner) input (sparse-scw-tmp-vec1 learner)) input tmp-float)
      (let ((v (+ sigma0 (aref tmp-float 0))))
        (declare (type (single-float 0.0) v))
        (let ((loss (- (* phi (sqrt v)) m)))
          (declare (type single-float loss))
          (when (> loss 0.0)
            (let ((alpha-sqrt-inner (+ (/ (* m m phi phi phi phi) 4.0) (* v phi phi zeta))))
              (declare (type (single-float 0.0) alpha-sqrt-inner))
              (let ((alpha (min C (max 0.0 (- (sqrt alpha-sqrt-inner) (* m psi))))))
                (declare (type single-float alpha))
                (let ((u-sqrt-inner (+ (* alpha alpha v v phi phi) (* 4.0 v))))
                  (declare (type (single-float 0.0) u-sqrt-inner))
                  (let ((u (let ((base (- (sqrt u-sqrt-inner) (* alpha v phi))))
                             (declare (type single-float base))
                             (/ (* base base) 4.0))))
                    (declare (type (single-float 0.0) u))
                    (let ((beta (/ (* alpha phi) (+ (sqrt u) (* v alpha phi)))))
                      (declare (type single-float beta))
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
                            (- sigma0 (* beta sigma0 sigma0))))))))))))))

;;; Logistic regression model (Sparse)

;; tmp-vec is pseudosparse-vector,

(defun logistic-regression-gradient-sparse!
    (training-label input-vector weight-vector bias C tmp-vec g-result g0-result)
  (declare (type single-float training-label bias C)
           (type sparse-vector input-vector)
           (type (simple-array single-float) weight-vector tmp-vec g-result)
           (type (simple-array single-float 1) g0-result)
           (optimize (speed 3) (safety 0)))
  (sf! input-vector weight-vector bias g0-result)
  (let ((sigmoid-val (sigmoid (* training-label (aref g0-result 0)))))
    (declare (type (single-float 0.0) sigmoid-val))
    ;; set gradient-vector to g-result
    (sps-v*n input-vector
             (* (- 1.0 sigmoid-val) (* -1.0 training-label))
             tmp-vec)
    (v*n weight-vector (* 2.0 C) g-result)
    (dps-v+ g-result tmp-vec (sparse-vector-index-vector input-vector) g-result)
    ;; return g0
    (setf (aref g0-result 0)
          (+ (* (- 1.0 sigmoid-val)
                (* -1.0 training-label))
             (* 2.0 C bias)))
    (values)))

;;; Sparse lr+sgd

(defstruct (sparse-lr+sgd (:constructor %make-sparse-lr+sgd))
  input-dimension weight bias
  ;; meta parameters
  C eta g tmp-vec tmp-float)

(defun make-sparse-lr+sgd (input-dimension C eta)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type C number)
  (check-type eta number)
  (let* ((C (coerce C 'single-float))
         (eta (coerce eta 'single-float)))
    (%make-sparse-lr+sgd
     :input-dimension input-dimension
     :weight (make-vec input-dimension 0.0)
     :bias 0.0
     :C C
     :eta eta
     :g (make-vec input-dimension 0.0)
     :tmp-vec (make-vec input-dimension 0.0)
     :tmp-float (make-vec 1 0.0))))

(define-learner sparse-lr+sgd (learner input training-label)
  (let ((weight (sparse-lr+sgd-weight learner))
        (bias (sparse-lr+sgd-bias learner))
        (C (sparse-lr+sgd-C learner))
        (eta (sparse-lr+sgd-eta learner))
        (tmp-vec (sparse-lr+sgd-tmp-vec learner))
        (g (sparse-lr+sgd-g learner))
        (tmp-float (sparse-lr+sgd-tmp-float learner)))
    (declare (type single-float bias C eta)
             (type (simple-array single-float) weight tmp-vec g)
             (type (simple-array single-float 1) tmp-float))
    ;; calc g (gradient)
    (logistic-regression-gradient-sparse! training-label input weight bias C tmp-vec g tmp-float)
    (v*n g eta g)
    (v- weight g weight)
    (setf (sparse-lr+sgd-bias learner) (- bias (* eta (aref tmp-float 0))))))

;;; Sparse lr+adam

(defstruct (sparse-lr+adam (:constructor %make-sparse-lr+adam)
                           (:print-object %print-sparse-lr+adam))
  input-dimension weight bias
  ;; meta parameters
  C alpha epsilon beta1 beta2
  ;; internal parameters
  g m v m0 v0 beta1^t beta2^t tmp-vec tmp-float)

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
  (check-type C number)
  (check-type alpha number)
  (check-type epsilon number)
  (check-type beta1 number)
  (check-type beta2 number)
  (assert (< 0.0 alpha))
  (assert (and (<= 0.0 beta1) (< beta1 1.0)))
  (assert (and (<= 0.0 beta2) (< beta2 1.0)))
  (%make-sparse-lr+adam
   :input-dimension input-dimension
   :weight (make-vec input-dimension 0.0)
   :bias 0.0
   :C (coerce C 'single-float)
   :alpha (coerce alpha 'single-float)
   :epsilon (coerce epsilon 'single-float)
   :beta1 (coerce beta1 'single-float)
   :beta2 (coerce beta2 'single-float)
   :g (make-vec input-dimension 0.0)
   :m (make-vec input-dimension 0.0)
   :v (make-vec input-dimension 0.0)
   :m0 0.0
   :v0 0.0
   :beta1^t beta1
   :beta2^t beta2
   :tmp-vec (make-vec input-dimension 0.0)
   :tmp-float (make-vec 1 0.0)))

(define-learner sparse-lr+adam (learner input training-label)
  (let ((weight (sparse-lr+adam-weight learner)) (bias (sparse-lr+adam-bias learner))
        (C (sparse-lr+adam-C learner)) (tmp-vec (sparse-lr+adam-tmp-vec learner)) (tmp-float (sparse-lr+adam-tmp-float learner))
        (g (sparse-lr+adam-g learner)) (g0 0.0)
        (m (sparse-lr+adam-m learner)) (m0 (sparse-lr+adam-m0 learner))
        (v (sparse-lr+adam-v learner)) (v0 (sparse-lr+adam-v0 learner))
        (alpha (sparse-lr+adam-alpha learner))
        (beta1 (sparse-lr+adam-beta1 learner)) (beta2 (sparse-lr+adam-beta2 learner))
        (beta1^t (sparse-lr+adam-beta1^t learner)) (beta2^t (sparse-lr+adam-beta2^t learner))
        (epsilon (sparse-lr+adam-epsilon learner)))
    (declare (type single-float bias C g0 m0 v0 alpha beta1 beta2 beta1^t beta2^t epsilon)
             (type (simple-array single-float) weight tmp-vec g m v)
             (type (simple-array single-float 1) tmp-float)
             (optimize (speed 3) (safety 0)))
    ;; calc g (gradient)
    (logistic-regression-gradient-sparse! training-label input weight bias C tmp-vec g tmp-float)
    (setf g0 (aref tmp-float 0))
    ;; update m_t from m_t-1
    (v*n m beta1 m)
    (v*n g (- 1.0 beta1) tmp-vec)
    (v+ m tmp-vec m)
    ;; calc g^2 (gradient^2)
    (v* g g g)
    ;; update v_t from v_t-1
    (v*n v beta2 v)
    (v*n g (- 1.0 beta2) tmp-vec)
    (v+ v tmp-vec v)
    ;; update m0 and v0
    (let ((new-m0 (+ (* beta1 m0) (* (- 1.0 beta1) g0)))
          (new-v0 (+ (* beta2 v0) (* (- 1.0 beta2) (* g0 g0))))
          (epsilon-coefficient-sqrt-inner (- 1.0 beta2^t)))
      (declare (type single-float new-m0)
               (type (single-float 0.0) new-v0 epsilon-coefficient-sqrt-inner))
      ;; update weight
      (let* ((epsilon-coefficient (sqrt epsilon-coefficient-sqrt-inner))
             (epsilon^ (* epsilon-coefficient epsilon))
             (alpha_t (* alpha (/ epsilon-coefficient (- 1.0 beta1^t)))))
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
         (learner test-data &key (quiet-p nil) (stream nil))
       (let* ((len (length test-data))
              (n-correct (count-if
                          (lambda (datum)
                            (let ((predict (,(intern (catstr (symbol-name learner-type) "-PREDICT"))
                                            learner (cdr datum))))
                              (format stream "~D~%" predict)
                              (= predict (car datum))))
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
    (loop for i from 0 below n-class do
      (setf (aref (one-vs-rest-learners-vector mulc) i)
            (apply (function-by-name (catstr "MAKE-" (symbol-name learner-type)))
                   (cons input-dimension learner-params))))
    mulc))

(defun one-vs-rest-predict (mulc input)
  (let ((max-f most-negative-single-float)
	(max-i 0))
    (loop for i from 0 below (one-vs-rest-n-class mulc) do
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
  (loop for i from 0 below (one-vs-rest-n-class mulc) do
    (if (= i training-label)
      (funcall (one-vs-rest-learner-update mulc)
               (svref (one-vs-rest-learners-vector mulc) i) input 1.0)
      (funcall (one-vs-rest-learner-update mulc)
               (svref (one-vs-rest-learners-vector mulc) i) input -1.0))))

(define-multi-class-learner-train/test-functions one-vs-rest)

;; for store/restore model with cl-store
(defun one-vs-rest-clear-functions-for-store (mulc)
  (setf (one-vs-rest-learner-weight   mulc) nil
        (one-vs-rest-learner-bias     mulc) nil
        (one-vs-rest-learner-update   mulc) nil
        (one-vs-rest-learner-activate mulc) nil))

(defun one-vs-rest-restore-functions (mulc)
  (let ((learner-type (type-of (aref (one-vs-rest-learners-vector mulc) 0))))
    (setf (one-vs-rest-learner-weight   mulc)
          (function-by-name (catstr (symbol-name learner-type) "-WEIGHT"))
          (one-vs-rest-learner-bias     mulc)
          (function-by-name (catstr (symbol-name learner-type) "-BIAS"))
          (one-vs-rest-learner-update   mulc)
          (function-by-name (catstr (symbol-name learner-type) "-UPDATE"))
          (one-vs-rest-learner-activate mulc)
          (if (sparse-symbol? learner-type)
              (lambda (input weight bias)
                (+ (ds-dot weight input) bias))
              (lambda (input weight bias)
                (+ (dot weight input) bias))))))

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
                :learner-update  (function-by-name (catstr (symbol-name learner-type) "-UPDATE"))
                :learner-predict (function-by-name (catstr (symbol-name learner-type) "-PREDICT")))))
    (loop for i from 0 below n-learner do
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
    (loop for k from 0 below (one-vs-one-n-class mulc) do
      (let ((cnt 0))
	;; negative
	(loop for i from 0 below k do
          ;; (format t "k: ~A, Negative, learner-index: ~A~%" k (index-of-learner k i (one-vs-one-n-class mulc)))
	  (if (< (funcall (one-vs-one-learner-predict mulc)
                          (svref (one-vs-one-learners-vector mulc)
                                 (index-of-learner k i (one-vs-one-n-class mulc))) input)
		 0.0)
	    (incf cnt)))
	;; positive
	(let ((start-index (sum-permutation (1- (one-vs-one-n-class mulc)) k)))
	  (loop for j from start-index to (+ start-index (- (1- (one-vs-one-n-class mulc)) k 1)) do
            ;; (format t "k: ~A, Positive, learner-index: ~A~%" k j)
	    (if (> (funcall (one-vs-one-learner-predict mulc)
                            (svref (one-vs-one-learners-vector mulc) j) input)
                   0.0)
	      (incf cnt))))
	(if (> cnt max-cnt)
	  (setf max-cnt cnt
		max-class k))))
    max-class))

;; training-label should be integer (0 ... K-1)
(defun one-vs-one-update (mulc input training-label)
  ;; negative
  (loop for i from 0 below training-label do
    ;; (format t "Negative. Index: ~A~%" (index-of-learner training-label i (one-vs-one-n-class mulc))) ;debug
    (funcall (one-vs-one-learner-update mulc)
             (svref (one-vs-one-learners-vector mulc)
                    (index-of-learner training-label i (one-vs-one-n-class mulc)))
             input -1.0))
  ;; positive
  (let ((start-index (sum-permutation (1- (one-vs-one-n-class mulc)) training-label)))
    (loop for j from start-index to (+ start-index (- (1- (one-vs-one-n-class mulc)) training-label 1)) do
      ;; (format t "Positive. Index: ~A~%" j) ;debug
      (funcall (one-vs-one-learner-update mulc)
               (svref (one-vs-one-learners-vector mulc) j)
               input 1.0))))

(define-multi-class-learner-train/test-functions one-vs-one)

;; for store/restore model with cl-store
(defun one-vs-one-clear-functions-for-store (mulc)
  (setf (one-vs-one-learner-update  mulc) nil
        (one-vs-one-learner-predict mulc) nil))

(defun one-vs-one-restore-functions (mulc)
  (let ((learner-type (type-of (aref (one-vs-one-learners-vector mulc) 0))))
    (setf (one-vs-one-learner-update  mulc)
          (function-by-name (catstr (symbol-name learner-type) "-UPDATE"))
          (one-vs-one-learner-predict mulc)
          (function-by-name (catstr (symbol-name learner-type) "-PREDICT")))))

;;; Save and restore models

(defun save (learner file-path)
  (typecase learner
    (one-vs-rest (one-vs-rest-clear-functions-for-store learner))
    (one-vs-one (one-vs-one-clear-functions-for-store learner)))
  (cl-store:store learner file-path)
  (typecase learner
      (one-vs-rest (one-vs-rest-restore-functions learner))
      (one-vs-one (one-vs-one-restore-functions learner)))
  learner)

(defun restore (file-path)
  (let ((learner (cl-store:restore file-path)))
    (typecase learner
      (one-vs-rest (one-vs-rest-restore-functions learner))
      (one-vs-one (one-vs-one-restore-functions learner)))
    learner))
