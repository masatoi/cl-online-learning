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
   :make-lr+ftrl :lr+ftrl-update :lr+ftrl-train :lr+ftrl-predict :lr+ftrl-test
   :make-sparse-perceptron :sparse-perceptron-update :sparse-perceptron-train
   :sparse-perceptron-predict :sparse-perceptron-test
   :make-sparse-arow :sparse-arow-update :sparse-arow-train :sparse-arow-predict :sparse-arow-test
   :make-sparse-scw :sparse-scw-update :sparse-scw-train :sparse-scw-predict :sparse-scw-test
   :make-sparse-lr+sgd :sparse-lr+sgd-update :sparse-lr+sgd-train :sparse-lr+sgd-predict :sparse-lr+sgd-test
   :make-sparse-lr+adam :sparse-lr+adam-update :sparse-lr+adam-train :sparse-lr+adam-predict :sparse-lr+adam-test
   :make-sparse-lr+ftrl :sparse-lr+ftrl-update :sparse-lr+ftrl-train
   :sparse-lr+ftrl-predict :sparse-lr+ftrl-test
   :make-softmax+ftrl :softmax+ftrl-update :softmax+ftrl-train
   :softmax+ftrl-predict :softmax+ftrl-test
   :make-sparse-softmax+ftrl :sparse-softmax+ftrl-update :sparse-softmax+ftrl-train
   :sparse-softmax+ftrl-predict :sparse-softmax+ftrl-test
   :make-one-vs-rest :one-vs-rest-update :one-vs-rest-train :one-vs-rest-predict :one-vs-rest-test
   :make-one-vs-one :one-vs-one-update :one-vs-one-train :one-vs-one-predict :one-vs-one-test
   :make-multiclass-arow :multiclass-arow-update :multiclass-arow-train
   :multiclass-arow-predict :multiclass-arow-test
   :make-sparse-multiclass-arow :sparse-multiclass-arow-update :sparse-multiclass-arow-train
   :sparse-multiclass-arow-predict :sparse-multiclass-arow-test
   ;; regression
   :make-rls :rls-update :rls-train :rls-predict :rls-test
   :make-sparse-rls :sparse-rls-update :sparse-rls-train :sparse-rls-predict :sparse-rls-test
   ; save/restore
   :save :restore
   :copy-learner))

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
    (typecase learner
      ;; A native multiclass learner's WEIGHT is a vector of K rows, so the
      ;; generic (LENGTH <TYPE>-WEIGHT) below would return K, not the dimension.
      (multiclass-arow (length (svref (multiclass-arow-weight learner) 0)))
      (sparse-multiclass-arow (length (svref (sparse-multiclass-arow-weight learner) 0)))
      (softmax+ftrl (length (svref (softmax+ftrl-weight learner) 0)))
      (sparse-softmax+ftrl (length (svref (sparse-softmax+ftrl-weight learner) 0)))
      (t (length (funcall (intern (catstr (symbol-name (type-of learner)) "-WEIGHT")
                                  :cl-online-learning)
                          learner))))))

(defun n-class-of (learner)
  (typecase learner
    (one-vs-one      (one-vs-one-n-class learner))
    (one-vs-rest     (one-vs-rest-n-class learner))
    (multiclass-arow (multiclass-arow-n-class learner))
    (sparse-multiclass-arow (sparse-multiclass-arow-n-class learner))
    (softmax+ftrl (softmax+ftrl-n-class learner))
    (sparse-softmax+ftrl (sparse-softmax+ftrl-n-class learner))
    (t 2)))

(defun sparse-learner? (learner)
  (typecase learner
    (one-vs-one  (sparse-symbol? (type-of (aref (one-vs-one-learners-vector learner) 0))))
    (one-vs-rest (sparse-symbol? (type-of (aref (one-vs-rest-learners-vector learner) 0))))
    (t (sparse-symbol? (type-of learner)))))

;;; Perceptron

(defstruct (perceptron (:constructor %make-perceptron) (:copier nil)
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

(defstruct (arow (:constructor  %make-arow) (:copier nil)
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

(defstruct (scw (:constructor  %make-scw) (:copier nil)
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
              ;; The 1/(v zeta) factor is Proposition 1's and was missing here until
              ;; 2026-08-02.  Without it every uncapped step is v*zeta times too large --
              ;; about 40x on a1a's first update -- which the C cap then hides: at eta 0.9
              ;; and C 0.1, 663 of 679 updates sat at the cap, so alpha was effectively a
              ;; constant and SCW-I's adaptive step size did nothing.  Do not "simplify"
              ;; it away; SCW-ALPHA-MATCHES-THE-PAPER-CLOSED-FORM pins it.
              (let ((alpha (min C (max 0.0 (/ (- (sqrt alpha-sqrt-inner) (* m psi))
                                              (* v zeta))))))
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

(defstruct (lr+sgd (:constructor %make-lr+sgd) (:copier nil))
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
(defstruct (lr+adam (:constructor %make-lr+adam) (:copier nil)
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

;;;; FTRL-Proximal
;;;;
;;;; McMahan, Holt, Sculley et al., "Ad Click Prediction: a View from the Trenches",
;;;; KDD 2013, Algorithm 1.  Logistic regression with per-coordinate adaptive learning
;;;; rates and L1 regularization strong enough to drive weights to exactly zero -- the
;;;; only learner here that produces a sparse model rather than merely accepting sparse
;;;; input.
;;;;
;;;; Per coordinate the state is (z_i, n_i) and the weight is derived, not stored:
;;;;
;;;;   w_i = 0                                              if |z_i| <= lambda1
;;;;   w_i = -(z_i - sgn(z_i) lambda1)
;;;;         / ((beta + sqrt(n_i))/alpha + lambda2)          otherwise
;;;;
;;;; then, for the coordinates the example touches,
;;;;
;;;;   g_i     = -y (1 - sigmoid(y (w.x + b))) x_i     [this repository's +-1 labels]
;;;;   sigma_i = (sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha
;;;;   z_i    += g_i - sigma_i w_i
;;;;   n_i    += g_i^2
;;;;
;;;; DEFINE-LEARNER's generated -PREDICT reads a WEIGHT slot, so w is materialised into
;;;; one.  That cache is exact because w_i is a pure function of (z_i, n_i), which change
;;;; only for the coordinates an update touches -- but ONLY if it is refreshed at the END
;;;; of the update.  See the ordering comment in LR+FTRL-UPDATE.
;;;;
;;;; w_i is also a function of ALPHA, BETA, LAMBDA1 and LAMBDA2 -- ordinary DEFSTRUCT
;;;; slots with SETF accessors, not just (z_i, n_i).  Treat those four as immutable after
;;;; construction: every coordinate's cached weight depends on all four, so changing one
;;;; post-construction desyncs the whole cache, not just the coordinates touched since.

(declaim (inline ftrl-weight-of))
(defun ftrl-weight-of (zi ni alpha beta lambda1 lambda2)
  "The FTRL-Proximal weight for one coordinate, derived from its (z, n) state."
  (declare (type single-float zi alpha beta lambda1 lambda2)
           (type (single-float 0.0) ni)
           (optimize (speed 3) (safety 0)))
  (if (<= (abs zi) lambda1)
    0.0
    (/ (- (- zi (* (if (> zi 0.0) 1.0 -1.0) lambda1)))
       (+ (/ (+ beta (sqrt ni)) alpha) lambda2))))

(defstruct (lr+ftrl (:constructor  %make-lr+ftrl) (:copier nil)
                    (:print-object %print-lr+ftrl))
  input-dimension weight bias
  ;; meta parameters
  alpha beta lambda1 lambda2
  ;; per-coordinate state, and the intercept's
  z n z0 n0)

(defun %print-lr+ftrl (obj stream)
  (format stream "#S(LR+FTRL~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A~%~T:NONZERO-WEIGHTS ~A/~A)"
          (lr+ftrl-input-dimension obj)
          (%vec-head (lr+ftrl-weight obj))
          (lr+ftrl-bias obj)
          (count-if-not #'zerop (lr+ftrl-weight obj))
          (length (lr+ftrl-weight obj))))

(defun make-lr+ftrl (input-dimension alpha beta lambda1 lambda2)
  (check-type input-dimension integer)
  (check-type alpha number)
  (check-type beta number)
  (check-type lambda1 number)
  (check-type lambda2 number)
  (assert (> input-dimension 0))
  (let ((alpha (coerce alpha 'single-float))
        (beta (coerce beta 'single-float))
        (lambda1 (coerce lambda1 'single-float))
        (lambda2 (coerce lambda2 'single-float)))
    ;; Assert AFTER coercion, not before.  A positive double such as 1d-50 satisfies
    ;; (< 0.0 alpha) yet underflows to exactly 0.0 as a single-float, and ALPHA divides
    ;; in every weight derivation -- the model would fill with NaN on the first update
    ;; and signal nothing.
    (assert (< 0.0 alpha))
    (assert (<= 0.0 beta))
    (assert (<= 0.0 lambda1))
    (assert (<= 0.0 lambda2))
    (%make-lr+ftrl
     :input-dimension input-dimension
     :weight (make-vec input-dimension 0.0)
     :bias 0.0
     :alpha alpha
     :beta beta
     :lambda1 lambda1
     :lambda2 lambda2
     :z (make-vec input-dimension 0.0)
     :n (make-vec input-dimension 0.0)
     :z0 0.0
     :n0 0.0)))

(define-learner lr+ftrl (learner input training-label)
  (let ((weight (lr+ftrl-weight learner))
        (z (lr+ftrl-z learner))
        (n (lr+ftrl-n learner))
        (alpha (lr+ftrl-alpha learner))
        (beta (lr+ftrl-beta learner))
        (lambda1 (lr+ftrl-lambda1 learner))
        (lambda2 (lr+ftrl-lambda2 learner)))
    (declare (type (simple-array single-float) weight z n)
             (type single-float alpha beta lambda1 lambda2))
    ;; 1. Predict with the cached weight.  It is already current: step 3 below
    ;;    refreshes it at the END of every update.  Refreshing at the START instead
    ;;    leaves the cache one update stale per coordinate -- measured at 89 of 123
    ;;    coordinates wrong on a1a, with accuracy indistinguishable either way.
    (let* ((fx (f input weight (lr+ftrl-bias learner)))
           (sigmoid-val (sigmoid (* training-label fx)))
           (gscale (* -1.0 training-label (- 1.0 sigmoid-val))))
      (declare (type single-float fx gscale)
               (type (single-float 0.0) sigmoid-val))
      ;; 2. Accumulate z and n, then 3. refresh the cache from the new state, fused
      ;;    into one pass.  The z update must read the w that produced the prediction
      ;;    above, so the refresh comes after it inside the loop body too.
      ;;    On a coordinate where INPUT is zero this is a no-op: g is 0, so z and n do
      ;;    not move and the refreshed w recomputes to the same value.  That is what
      ;;    makes the dense and sparse variants agree exactly.
      (dovec weight i
        (let* ((gi (* gscale (aref input i)))
               (ni (aref n i))
               (new-ni (+ ni (* gi gi))))
          (declare (type single-float gi)
                   (type (single-float 0.0) ni new-ni))
          (incf (aref z i) (- gi (* (/ (- (sqrt new-ni) (sqrt ni)) alpha)
                                    (aref weight i))))
          (setf (aref n i) new-ni
                (aref weight i)
                (ftrl-weight-of (aref z i) new-ni alpha beta lambda1 lambda2))))
      ;; The intercept is an always-1 feature and is deliberately NOT L1-regularised:
      ;; an L1-zeroed intercept would force the boundary through the origin.
      (let* ((n0 (lr+ftrl-n0 learner))
             (new-n0 (+ n0 (* gscale gscale))))
        (declare (type (single-float 0.0) n0 new-n0))
        (incf (lr+ftrl-z0 learner)
              (- gscale (* (/ (- (sqrt new-n0) (sqrt n0)) alpha)
                           (lr+ftrl-bias learner))))
        (setf (lr+ftrl-n0 learner) new-n0
              (lr+ftrl-bias learner)
              (ftrl-weight-of (lr+ftrl-z0 learner) new-n0 alpha beta 0.0 lambda2))))))

;;;; Sparse version learners ;;;;

;;; Sparse Perceptron

(defstruct (sparse-perceptron (:constructor %make-sparse-perceptron) (:copier nil)
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

(defstruct (sparse-arow (:constructor  %make-sparse-arow) (:copier nil)
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

(defstruct (sparse-scw (:constructor  %make-sparse-scw) (:copier nil)
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
              ;; The 1/(v zeta) factor is Proposition 1's and was missing here until
              ;; 2026-08-02.  Without it every uncapped step is v*zeta times too large --
              ;; about 40x on a1a's first update -- which the C cap then hides: at eta 0.9
              ;; and C 0.1, 663 of 679 updates sat at the cap, so alpha was effectively a
              ;; constant and SCW-I's adaptive step size did nothing.  Do not "simplify"
              ;; it away; SPARSE-SCW-ALPHA-MATCHES-THE-PAPER-CLOSED-FORM pins it.
              (let ((alpha (min C (max 0.0 (/ (- (sqrt alpha-sqrt-inner) (* m psi))
                                              (* v zeta))))))
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

(defstruct (sparse-lr+sgd (:constructor %make-sparse-lr+sgd) (:copier nil))
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

(defstruct (sparse-lr+adam (:constructor %make-sparse-lr+adam) (:copier nil)
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

;;; Sparse FTRL-Proximal
;;;
;;; Identical arithmetic to LR+FTRL; only the traversal differs.  The update walks the
;;; input's index-vector and nothing else, so it is genuinely O(nnz) -- unlike
;;; SPARSE-LR+ADAM above, whose moment updates sweep the full dimension and are sparse
;;; only in the gradient.  Z, N and WEIGHT are full-length dense arrays read and written
;;; only at the active indices; untouched coordinates keep z = 0, hence w = 0, which is
;;; their initial value.

(defstruct (sparse-lr+ftrl (:constructor  %make-sparse-lr+ftrl) (:copier nil)
                           (:print-object %print-sparse-lr+ftrl))
  input-dimension weight bias
  ;; meta parameters
  alpha beta lambda1 lambda2
  ;; per-coordinate state, and the intercept's
  z n z0 n0)

(defun %print-sparse-lr+ftrl (obj stream)
  (format stream "#S(SPARSE-LR+FTRL~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A~%~T:NONZERO-WEIGHTS ~A/~A)"
          (sparse-lr+ftrl-input-dimension obj)
          (%vec-head (sparse-lr+ftrl-weight obj))
          (sparse-lr+ftrl-bias obj)
          (count-if-not #'zerop (sparse-lr+ftrl-weight obj))
          (length (sparse-lr+ftrl-weight obj))))

(defun make-sparse-lr+ftrl (input-dimension alpha beta lambda1 lambda2)
  (check-type input-dimension integer)
  (check-type alpha number)
  (check-type beta number)
  (check-type lambda1 number)
  (check-type lambda2 number)
  (assert (> input-dimension 0))
  (let ((alpha (coerce alpha 'single-float))
        (beta (coerce beta 'single-float))
        (lambda1 (coerce lambda1 'single-float))
        (lambda2 (coerce lambda2 'single-float)))
    ;; Assert AFTER coercion, not before.  A positive double such as 1d-50 satisfies
    ;; (< 0.0 alpha) yet underflows to exactly 0.0 as a single-float, and ALPHA divides
    ;; in every weight derivation -- the model would fill with NaN on the first update
    ;; and signal nothing.
    (assert (< 0.0 alpha))
    (assert (<= 0.0 beta))
    (assert (<= 0.0 lambda1))
    (assert (<= 0.0 lambda2))
    (%make-sparse-lr+ftrl
     :input-dimension input-dimension
     :weight (make-vec input-dimension 0.0)
     :bias 0.0
     :alpha alpha
     :beta beta
     :lambda1 lambda1
     :lambda2 lambda2
     :z (make-vec input-dimension 0.0)
     :n (make-vec input-dimension 0.0)
     :z0 0.0
     :n0 0.0)))

(define-learner sparse-lr+ftrl (learner input training-label)
  (let ((weight (sparse-lr+ftrl-weight learner))
        (z (sparse-lr+ftrl-z learner))
        (n (sparse-lr+ftrl-n learner))
        (alpha (sparse-lr+ftrl-alpha learner))
        (beta (sparse-lr+ftrl-beta learner))
        (lambda1 (sparse-lr+ftrl-lambda1 learner))
        (lambda2 (sparse-lr+ftrl-lambda2 learner))
        (index-vector (sparse-vector-index-vector input))
        (value-vector (sparse-vector-value-vector input)))
    (declare (type (simple-array single-float) weight z n value-vector)
             (type (simple-array fixnum) index-vector)
             (type single-float alpha beta lambda1 lambda2))
    ;; 1. Predict with the cached weight, which step 3 keeps current.
    (let* ((fx (sf input weight (sparse-lr+ftrl-bias learner)))
           (sigmoid-val (sigmoid (* training-label fx)))
           (gscale (* -1.0 training-label (- 1.0 sigmoid-val))))
      (declare (type single-float fx gscale)
               (type (single-float 0.0) sigmoid-val))
      ;; 2. Accumulate z and n, then 3. refresh the cache, fused into one O(nnz) pass.
      ;;    The z update must read the w that produced the prediction, so the refresh
      ;;    comes after it inside the loop body.
      (dosvec input k
        (let* ((i (aref index-vector k))
               (gi (* gscale (aref value-vector k)))
               (ni (aref n i))
               (new-ni (+ ni (* gi gi))))
          (declare (type fixnum i)
                   (type single-float gi)
                   (type (single-float 0.0) ni new-ni))
          (incf (aref z i) (- gi (* (/ (- (sqrt new-ni) (sqrt ni)) alpha)
                                    (aref weight i))))
          (setf (aref n i) new-ni
                (aref weight i)
                (ftrl-weight-of (aref z i) new-ni alpha beta lambda1 lambda2))))
      ;; The intercept is an always-1 feature and is deliberately NOT L1-regularised.
      (let* ((n0 (sparse-lr+ftrl-n0 learner))
             (new-n0 (+ n0 (* gscale gscale))))
        (declare (type (single-float 0.0) n0 new-n0))
        (incf (sparse-lr+ftrl-z0 learner)
              (- gscale (* (/ (- (sqrt new-n0) (sqrt n0)) alpha)
                           (sparse-lr+ftrl-bias learner))))
        (setf (sparse-lr+ftrl-n0 learner) new-n0
              (sparse-lr+ftrl-bias learner)
              (ftrl-weight-of (sparse-lr+ftrl-z0 learner) new-n0
                              alpha beta 0.0 lambda2))))))

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

(defstruct (one-vs-rest (:constructor  %make-one-vs-rest) (:copier nil)
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

(defstruct (one-vs-one (:constructor  %make-one-vs-one) (:copier nil)
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

;;;; Multiclass AROW
;;;;
;;;; The top-1 version of multi-class AROW, Figure 3 of Crammer, Kulesza &
;;;; Dredze, "Adaptive regularization of weight vectors", Machine Learning
;;;; 91(2):155-187, 2013.  Unlike ONE-VS-REST and ONE-VS-ONE this is not a
;;;; wrapper: it is a single learner holding K weight vectors, updated from the
;;;; margin between the true class and its closest competitor.
;;;;
;;;; Under the paper's 1-of-K feature map with a diagonal covariance per class,
;;;; Figure 3 reduces to (writing s for the closest competitor of the true
;;;; class y, and gamma for the paper's r):
;;;;
;;;;   m = (mu_y . x + b_y) - (mu_s . x + b_s)
;;;;   v = (x' Sigma_y x + sigma0_y) + (x' Sigma_s x + sigma0_s)
;;;;   beta = 1 / (v + gamma)        alpha = (1 - m) * beta
;;;;   mu_y    += alpha * Sigma_y x       mu_s    -= alpha * Sigma_s x
;;;;   b_y     += alpha * sigma0_y        b_s     -= alpha * sigma0_s
;;;;   Sigma_y -= beta * (Sigma_y x)^2    Sigma_s -= beta * (Sigma_s x)^2
;;;;   sigma0_y -= beta * sigma0_y^2      sigma0_s -= beta * sigma0_s^2
;;;;
;;;; which is exactly the binary AROW update body generalized to two classes, so
;;;; every existing CLOL.VECTOR operator applies and none had to be added.

(defun %vec-head (vec &optional (n 10))
  "Return the first N elements of VEC, or VEC itself when it is no longer than N."
  (if (> (length vec) n) (subseq vec 0 n) vec))

(defun %make-weight-vectors (n-class input-dimension initial-element)
  "Return a SIMPLE-VECTOR of N-CLASS fresh INPUT-DIMENSION vectors of INITIAL-ELEMENT.
Each row is a plain (SIMPLE-ARRAY SINGLE-FLOAT), identical in type to every other
learner's weight, which is what lets the existing vector operators apply per row."
  (let ((rows (make-array n-class)))
    (dotimes (i n-class rows)
      (setf (svref rows i) (make-vec input-dimension initial-element)))))

(defstruct (multiclass-arow (:constructor  %make-multiclass-arow) (:copier nil)
                            (:print-object %print-multiclass-arow))
  input-dimension n-class weight bias
  gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-vec3)

(defun %print-multiclass-arow (obj stream)
  (format stream "#S(MULTICLASS-AROW~%~T:INPUT-DIMENSION ~A~%~T:N-CLASS ~A~%~T:WEIGHT #(~A ...)~%~T:BIAS ~A~%~T:GAMMA ~A~%~T:SIGMA #(~A ...)~%~T:SIGMA0 ~A)"
          (multiclass-arow-input-dimension obj)
          (multiclass-arow-n-class obj)
          (%vec-head (svref (multiclass-arow-weight obj) 0))
          (%vec-head (multiclass-arow-bias obj))
          (multiclass-arow-gamma obj)
          (%vec-head (svref (multiclass-arow-sigma obj) 0))
          (%vec-head (multiclass-arow-sigma0 obj))))

(defun make-multiclass-arow (input-dimension n-class gamma)
  (check-type input-dimension integer)
  (check-type n-class integer)
  (check-type gamma number)
  (assert (> input-dimension 0))
  ;; N-CLASS 2 would make N-CLASS-OF return 2, putting CLOL-PREDICT on the binary
  ;; label path and silently misreading the dataset.  Same bound as
  ;; MAKE-ONE-VS-REST and MAKE-ONE-VS-ONE.
  (assert (> n-class 2))
  (%make-multiclass-arow
   :input-dimension input-dimension
   :n-class n-class
   :weight (%make-weight-vectors n-class input-dimension 0.0) ; mu
   :bias (make-vec n-class 0.0)                               ; mu0
   :gamma (coerce gamma 'single-float)
   :sigma (%make-weight-vectors n-class input-dimension 1.0)
   :sigma0 (make-vec n-class 1.0)
   :tmp-vec1 (make-vec input-dimension 0.0)
   :tmp-vec2 (make-vec input-dimension 0.0)
   :tmp-vec3 (make-vec input-dimension 0.0)))

(defun multiclass-arow-predict (learner input)
  (declare (type multiclass-arow learner)
           (type (simple-array single-float) input)
           (optimize (speed 3) (safety 0)))
  (let ((weight (multiclass-arow-weight learner))
        (bias (multiclass-arow-bias learner))
        (n-class (multiclass-arow-n-class learner))
        (max-f most-negative-single-float)
        (max-i 0))
    (declare (type simple-vector weight)
             (type (simple-array single-float) bias)
             (type fixnum n-class max-i)
             (type single-float max-f))
    ;; Strict >, so the lowest index wins a tie -- same rule as
    ;; ONE-VS-REST-PREDICT, and what makes the golden values reproducible.
    (loop for i of-type fixnum from 0 below n-class do
      (let ((fi (+ (dot (the (simple-array single-float) (svref weight i)) input)
                   (aref bias i))))
        (declare (type single-float fi))
        (when (> fi max-f)
          (setf max-f fi
                max-i i))))
    max-i))

;; training-label should be an integer class index (0 ... K-1)
;; This range is not checked under (safety 0): an out-of-range label corrupts
;; memory (an unchecked heap write past WEIGHT/BIAS) rather than signalling.
(defun multiclass-arow-update (learner input training-label)
  (declare (type multiclass-arow learner)
           (type (simple-array single-float) input)
           (type fixnum training-label)
           (optimize (speed 3) (safety 0)))
  (let ((weight (multiclass-arow-weight learner))
        (bias (multiclass-arow-bias learner))
        (sigma (multiclass-arow-sigma learner))
        (sigma0 (multiclass-arow-sigma0 learner))
        (n-class (multiclass-arow-n-class learner))
        (f-y 0.0)
        (f-s most-negative-single-float)
        (s 0))
    (declare (type simple-vector weight sigma)
             (type (simple-array single-float) bias sigma0)
             (type fixnum n-class s)
             (type single-float f-y f-s))
    ;; One pass: the true class's score, and the best-scoring competitor.
    (loop for i of-type fixnum from 0 below n-class do
      (let ((fi (+ (dot (the (simple-array single-float) (svref weight i)) input)
                   (aref bias i))))
        (declare (type single-float fi))
        (if (= i training-label)
          (setf f-y fi)
          (when (> fi f-s)
            (setf f-s fi
                  s i)))))
    (let ((m (- f-y f-s)))
      (declare (type single-float m))
      (when (< m 1.0)
        (let ((weight-y (svref weight training-label))
              (weight-s (svref weight s))
              (sigma-y (svref sigma training-label))
              (sigma-s (svref sigma s))
              (tmp-vec1 (multiclass-arow-tmp-vec1 learner))
              (tmp-vec2 (multiclass-arow-tmp-vec2 learner))
              (tmp-vec3 (multiclass-arow-tmp-vec3 learner))
              (sigma0-y (aref sigma0 training-label))
              (sigma0-s (aref sigma0 s)))
          (declare (type (simple-array single-float)
                         weight-y weight-s sigma-y sigma-s tmp-vec1 tmp-vec2 tmp-vec3)
                   (type single-float sigma0-y sigma0-s))
          ;; Both Sigma_k x are needed before the confidence v can be formed, and
          ;; both must survive the weight update intact because the sigma update
          ;; squares them -- hence three scratch vectors rather than two.
          (v* sigma-y input tmp-vec1)
          (v* sigma-s input tmp-vec2)
          (let* ((v (+ (dot tmp-vec1 input) (dot tmp-vec2 input) sigma0-y sigma0-s))
                 (beta (/ 1.0 (+ v (multiclass-arow-gamma learner))))
                 ;; No MAX(0, ...) needed: the M < 1.0 guard makes 1 - m positive.
                 (alpha (* (- 1.0 m) beta)))
            (declare (type single-float v beta alpha))
            ;; Update weight
            (v+ weight-y (v*n tmp-vec1 alpha tmp-vec3) weight-y)
            (v- weight-s (v*n tmp-vec2 alpha tmp-vec3) weight-s)
            ;; Update bias
            (setf (aref bias training-label) (+ (aref bias training-label)
                                                (* alpha sigma0-y))
                  (aref bias s)              (- (aref bias s)
                                                (* alpha sigma0-s)))
            ;; Update sigma
            (v- sigma-y (v*n (v* tmp-vec1 tmp-vec1 tmp-vec1) beta tmp-vec1) sigma-y)
            (v- sigma-s (v*n (v* tmp-vec2 tmp-vec2 tmp-vec2) beta tmp-vec2) sigma-s)
            ;; Update sigma0
            (setf (aref sigma0 training-label) (- sigma0-y (* beta sigma0-y sigma0-y))
                  (aref sigma0 s)              (- sigma0-s (* beta sigma0-s sigma0-s))))))))
  learner)

(define-multi-class-learner-train/test-functions multiclass-arow)

;;; Sparse Multiclass AROW
;;;
;;; Identical arithmetic to MULTICLASS-AROW; only the traversal differs.  SIGMA
;;; rows and the three scratch vectors are pseudosparse: full-length dense
;;; arrays read and written only at the indices in the input's INDEX-VECTOR, so
;;; an update stays O(nnz) while the storage stays a plain dense array.  Values
;;; left in a scratch vector outside the current index set are stale and are
;;; never read, which is what makes reusing one scratch across data points safe.

(defstruct (sparse-multiclass-arow (:constructor  %make-sparse-multiclass-arow) (:copier nil)
                                   (:print-object %print-sparse-multiclass-arow))
  input-dimension n-class weight bias
  gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-vec3)

(defun %print-sparse-multiclass-arow (obj stream)
  (format stream "#S(SPARSE-MULTICLASS-AROW~%~T:INPUT-DIMENSION ~A~%~T:N-CLASS ~A~%~T:WEIGHT #(~A ...)~%~T:BIAS ~A~%~T:GAMMA ~A~%~T:SIGMA #(~A ...)~%~T:SIGMA0 ~A)"
          (sparse-multiclass-arow-input-dimension obj)
          (sparse-multiclass-arow-n-class obj)
          (%vec-head (svref (sparse-multiclass-arow-weight obj) 0))
          (%vec-head (sparse-multiclass-arow-bias obj))
          (sparse-multiclass-arow-gamma obj)
          (%vec-head (svref (sparse-multiclass-arow-sigma obj) 0))
          (%vec-head (sparse-multiclass-arow-sigma0 obj))))

(defun make-sparse-multiclass-arow (input-dimension n-class gamma)
  (check-type input-dimension integer)
  (check-type n-class integer)
  (check-type gamma number)
  (assert (> input-dimension 0))
  (assert (> n-class 2))
  (%make-sparse-multiclass-arow
   :input-dimension input-dimension
   :n-class n-class
   :weight (%make-weight-vectors n-class input-dimension 0.0) ; mu
   :bias (make-vec n-class 0.0)                               ; mu0
   :gamma (coerce gamma 'single-float)
   :sigma (%make-weight-vectors n-class input-dimension 1.0)
   :sigma0 (make-vec n-class 1.0)
   :tmp-vec1 (make-vec input-dimension 0.0)
   :tmp-vec2 (make-vec input-dimension 0.0)
   :tmp-vec3 (make-vec input-dimension 0.0)))

(defun sparse-multiclass-arow-predict (learner input)
  (declare (type sparse-multiclass-arow learner)
           (type sparse-vector input)
           (optimize (speed 3) (safety 0)))
  (let ((weight (sparse-multiclass-arow-weight learner))
        (bias (sparse-multiclass-arow-bias learner))
        (n-class (sparse-multiclass-arow-n-class learner))
        (max-f most-negative-single-float)
        (max-i 0))
    (declare (type simple-vector weight)
             (type (simple-array single-float) bias)
             (type fixnum n-class max-i)
             (type single-float max-f))
    (loop for i of-type fixnum from 0 below n-class do
      (let ((fi (+ (ds-dot (the (simple-array single-float) (svref weight i)) input)
                   (aref bias i))))
        (declare (type single-float fi))
        (when (> fi max-f)
          (setf max-f fi
                max-i i))))
    max-i))

;; training-label should be an integer class index (0 ... K-1)
;; This range is not checked under (safety 0): an out-of-range label corrupts
;; memory (an unchecked heap write past WEIGHT/BIAS) rather than signalling.
(defun sparse-multiclass-arow-update (learner input training-label)
  (declare (type sparse-multiclass-arow learner)
           (type sparse-vector input)
           (type fixnum training-label)
           (optimize (speed 3) (safety 0)))
  (let ((weight (sparse-multiclass-arow-weight learner))
        (bias (sparse-multiclass-arow-bias learner))
        (sigma (sparse-multiclass-arow-sigma learner))
        (sigma0 (sparse-multiclass-arow-sigma0 learner))
        (n-class (sparse-multiclass-arow-n-class learner))
        (index-vector (sparse-vector-index-vector input))
        (f-y 0.0)
        (f-s most-negative-single-float)
        (s 0))
    (declare (type simple-vector weight sigma)
             (type (simple-array single-float) bias sigma0)
             (type (simple-array fixnum) index-vector)
             (type fixnum n-class s)
             (type single-float f-y f-s))
    (loop for i of-type fixnum from 0 below n-class do
      (let ((fi (+ (ds-dot (the (simple-array single-float) (svref weight i)) input)
                   (aref bias i))))
        (declare (type single-float fi))
        (if (= i training-label)
          (setf f-y fi)
          (when (> fi f-s)
            (setf f-s fi
                  s i)))))
    (let ((m (- f-y f-s)))
      (declare (type single-float m))
      (when (< m 1.0)
        (let ((weight-y (svref weight training-label))
              (weight-s (svref weight s))
              (sigma-y (svref sigma training-label))
              (sigma-s (svref sigma s))
              (tmp-vec1 (sparse-multiclass-arow-tmp-vec1 learner))
              (tmp-vec2 (sparse-multiclass-arow-tmp-vec2 learner))
              (tmp-vec3 (sparse-multiclass-arow-tmp-vec3 learner))
              (sigma0-y (aref sigma0 training-label))
              (sigma0-s (aref sigma0 s)))
          (declare (type (simple-array single-float)
                         weight-y weight-s sigma-y sigma-s tmp-vec1 tmp-vec2 tmp-vec3)
                   (type single-float sigma0-y sigma0-s))
          (ds-v* sigma-y input tmp-vec1)
          (ds-v* sigma-s input tmp-vec2)
          (let* ((v (+ (ds-dot tmp-vec1 input) (ds-dot tmp-vec2 input) sigma0-y sigma0-s))
                 (beta (/ 1.0 (+ v (sparse-multiclass-arow-gamma learner))))
                 (alpha (* (- 1.0 m) beta)))
            (declare (type single-float v beta alpha))
            ;; Update weight
            (dps-v+ weight-y (ps-v*n tmp-vec1 alpha index-vector tmp-vec3)
                    index-vector weight-y)
            (dps-v- weight-s (ps-v*n tmp-vec2 alpha index-vector tmp-vec3)
                    index-vector weight-s)
            ;; Update bias
            (setf (aref bias training-label) (+ (aref bias training-label)
                                                (* alpha sigma0-y))
                  (aref bias s)              (- (aref bias s)
                                                (* alpha sigma0-s)))
            ;; Update sigma
            (dps-v- sigma-y (ps-v*n (dps-v* tmp-vec1 tmp-vec1 index-vector tmp-vec1)
                                    beta index-vector tmp-vec1)
                    index-vector sigma-y)
            (dps-v- sigma-s (ps-v*n (dps-v* tmp-vec2 tmp-vec2 index-vector tmp-vec2)
                                    beta index-vector tmp-vec2)
                    index-vector sigma-s)
            ;; Update sigma0
            (setf (aref sigma0 training-label) (- sigma0-y (* beta sigma0-y sigma0-y))
                  (aref sigma0 s)              (- sigma0-s (* beta sigma0-s sigma0-s))))))))
  learner)

(define-multi-class-learner-train/test-functions sparse-multiclass-arow)

;;;; Softmax regression with FTRL-Proximal
;;;;
;;;; FTRL-Proximal is an optimizer, not a loss: the per-coordinate (z, n) state, the L1
;;;; soft-threshold, the adaptive learning rate and the materialised weight cache are all
;;;; LR+FTRL's, reused unchanged down to FTRL-WEIGHT-OF.  Only the gradient differs.
;;;;
;;;; McMahan et al. cover binary logistic regression only, so the multiclass loss is a
;;;; choice rather than a transcription.  This is the multinomial logistic (softmax)
;;;; loss, the standard multiclass extension of the loss LR+FTRL already optimizes:
;;;;
;;;;   f_k   = w_k . x + b_k                for every class k
;;;;   p     = softmax(f)
;;;;   loss  = -log p_y
;;;;   g_k,i = (p_k - [k = y]) x_i          [k = y] is 1 for the true class, else 0
;;;;
;;;; Every class gets a non-zero gradient on every example, so an update touches all K
;;;; rows at the input's non-zero coordinates.  At K = 2 this reduces to g_y = p_y - 1,
;;;; which is exactly LR+FTRL's GSCALE.
;;;;
;;;; EXP overflows in single-float above roughly 88 and weights are unbounded, so the
;;;; softmax subtracts the maximum score before exponentiating.  After that subtraction
;;;; the largest exponent is 0, so nothing overflows, and the denominator is at least 1,
;;;; so there is no division by zero.  Terms that underflow to 0 contribute 0, which is
;;;; correct.  This is the first place in this file that needs such a guard -- SIGMOID
;;;; computes 1/(1 + exp(-x)), where a large positive x merely underflows harmlessly.
;;;;
;;;; ALPHA, BETA, LAMBDA1 and LAMBDA2 must be treated as immutable after construction:
;;;; every coordinate's weight is derived from them, so changing one silently
;;;; invalidates the entire WEIGHT cache.

(defstruct (softmax+ftrl (:constructor  %make-softmax+ftrl) (:copier nil)
                         (:print-object %print-softmax+ftrl))
  input-dimension n-class weight bias
  ;; meta parameters
  alpha beta lambda1 lambda2
  ;; per-coordinate state, the intercept's, and the score/probability scratch
  z n z0 n0 tmp-p)

(defun %print-softmax+ftrl (obj stream)
  (format stream "#S(SOFTMAX+FTRL~%~T:INPUT-DIMENSION ~A~%~T:N-CLASS ~A~%~T:WEIGHT #(~A ...)~%~T:BIAS ~A~%~T:NONZERO-WEIGHTS ~A/~A)"
          (softmax+ftrl-input-dimension obj)
          (softmax+ftrl-n-class obj)
          (%vec-head (svref (softmax+ftrl-weight obj) 0))
          (%vec-head (softmax+ftrl-bias obj))
          (reduce #'+ (softmax+ftrl-weight obj) :initial-value 0
                  :key (lambda (row) (count-if-not #'zerop row)))
          (* (softmax+ftrl-n-class obj) (softmax+ftrl-input-dimension obj))))

(defun make-softmax+ftrl (input-dimension n-class alpha beta lambda1 lambda2)
  (check-type input-dimension integer)
  (check-type n-class integer)
  (check-type alpha number)
  (check-type beta number)
  (check-type lambda1 number)
  (check-type lambda2 number)
  (assert (> input-dimension 0))
  ;; With N-CLASS 2, N-CLASS-OF returns 2, CLOL-PREDICT takes the binary label path and
  ;; silently misreads the dataset.  Same bound as MAKE-MULTICLASS-AROW.
  (assert (> n-class 2))
  (let ((alpha (coerce alpha 'single-float))
        (beta (coerce beta 'single-float))
        (lambda1 (coerce lambda1 'single-float))
        (lambda2 (coerce lambda2 'single-float)))
    ;; Assert AFTER coercion, not before.  A positive double such as 1d-50 satisfies
    ;; (< 0.0 alpha) yet underflows to exactly 0.0 as a single-float, and ALPHA divides
    ;; in every weight derivation -- the model would fill with NaN and signal nothing.
    (assert (< 0.0 alpha))
    (assert (<= 0.0 beta))
    (assert (<= 0.0 lambda1))
    (assert (<= 0.0 lambda2))
    (%make-softmax+ftrl
     :input-dimension input-dimension
     :n-class n-class
     :weight (%make-weight-vectors n-class input-dimension 0.0)
     :bias (make-vec n-class 0.0)
     :alpha alpha
     :beta beta
     :lambda1 lambda1
     :lambda2 lambda2
     :z (%make-weight-vectors n-class input-dimension 0.0)
     :n (%make-weight-vectors n-class input-dimension 0.0)
     :z0 (make-vec n-class 0.0)
     :n0 (make-vec n-class 0.0)
     :tmp-p (make-vec n-class 0.0))))

(defun softmax+ftrl-predict (learner input)
  (declare (type softmax+ftrl learner)
           (type (simple-array single-float) input)
           (optimize (speed 3) (safety 0)))
  (let ((weight (softmax+ftrl-weight learner))
        (bias (softmax+ftrl-bias learner))
        (n-class (softmax+ftrl-n-class learner))
        (max-f most-negative-single-float)
        (max-i 0))
    (declare (type simple-vector weight)
             (type (simple-array single-float) bias)
             (type fixnum n-class max-i)
             (type single-float max-f))
    ;; The softmax is monotone, so the argmax of the scores is the argmax of the
    ;; probabilities -- normalising here would be wasted work.  Strict > means the lowest
    ;; index wins a tie, as in ONE-VS-REST-PREDICT and MULTICLASS-AROW-PREDICT.
    (loop for k of-type fixnum from 0 below n-class do
      (let ((fk (+ (dot (the (simple-array single-float) (svref weight k)) input)
                   (aref bias k))))
        (declare (type single-float fk))
        (when (> fk max-f)
          (setf max-f fk
                max-i k))))
    max-i))

;; training-label should be an integer class index (0 ... K-1).  The range is NOT
;; checked. Unlike MULTICLASS-AROW-UPDATE, which indexes by the label and faults on a bad one, this
;; body only compares against it, so an out-of-range label causes no fault -- just a wrong gradient:
;; every class is treated as incorrect, so all K rows are pushed down with no positive target.
;; Realistic cause: raw LIBSVM labels instead of :multiclass-p t.
(defun softmax+ftrl-update (learner input training-label)
  (declare (type softmax+ftrl learner)
           (type (simple-array single-float) input)
           (type fixnum training-label)
           (optimize (speed 3) (safety 0)))
  (let ((weight (softmax+ftrl-weight learner))
        (bias (softmax+ftrl-bias learner))
        (z (softmax+ftrl-z learner))
        (n (softmax+ftrl-n learner))
        (z0 (softmax+ftrl-z0 learner))
        (n0 (softmax+ftrl-n0 learner))
        (p (softmax+ftrl-tmp-p learner))
        (n-class (softmax+ftrl-n-class learner))
        (alpha (softmax+ftrl-alpha learner))
        (beta (softmax+ftrl-beta learner))
        (lambda1 (softmax+ftrl-lambda1 learner))
        (lambda2 (softmax+ftrl-lambda2 learner)))
    (declare (type simple-vector weight z n)
             (type (simple-array single-float) bias z0 n0 p)
             (type fixnum n-class)
             (type single-float alpha beta lambda1 lambda2))
    ;; 1. Score every class into P, tracking the maximum.
    (let ((max-f most-negative-single-float))
      (declare (type single-float max-f))
      (loop for k of-type fixnum from 0 below n-class do
        (let ((fk (+ (dot (the (simple-array single-float) (svref weight k)) input)
                     (aref bias k))))
          (declare (type single-float fk))
          (setf (aref p k) fk)
          (when (> fk max-f) (setf max-f fk))))
      ;; 2. Turn P into probabilities.  Subtracting MAX-F before EXP is what keeps this
      ;;    from overflowing: afterwards the largest exponent is 0, so no term exceeds
      ;;    1.0, and SUM is at least 1.0 so the division is safe.
      (let ((sum 0.0))
        (declare (type single-float sum))
        (loop for k of-type fixnum from 0 below n-class do
          (let ((e (exp (- (aref p k) max-f))))
            (declare (type single-float e))
            (setf (aref p k) e)
            (incf sum e)))
        (loop for k of-type fixnum from 0 below n-class do
          (setf (aref p k) (/ (aref p k) sum)))))
    ;; 3. One FTRL step per class.  Within each row the z update must read the weight
    ;;    that produced the score above, so the cache refresh comes after it -- the same
    ;;    ordering LR+FTRL depends on, and just as load-bearing here.
    (loop for k of-type fixnum from 0 below n-class do
      (let ((gscale (- (aref p k) (if (= k training-label) 1.0 0.0)))
            (weight-k (svref weight k))
            (z-k (svref z k))
            (n-k (svref n k)))
        (declare (type single-float gscale)
                 (type (simple-array single-float) weight-k z-k n-k))
        (dovec weight-k i
          (let* ((gi (* gscale (aref input i)))
                 (ni (aref n-k i))
                 (new-ni (+ ni (* gi gi))))
            (declare (type single-float gi)
                     (type (single-float 0.0) ni new-ni))
            (incf (aref z-k i) (- gi (* (/ (- (sqrt new-ni) (sqrt ni)) alpha)
                                        (aref weight-k i))))
            (setf (aref n-k i) new-ni
                  (aref weight-k i)
                  (ftrl-weight-of (aref z-k i) new-ni alpha beta lambda1 lambda2))))
        ;; The intercept is an always-1 feature and is deliberately NOT L1-regularised.
        (let* ((nk0 (aref n0 k))
               (new-n0 (+ nk0 (* gscale gscale))))
          (declare (type (single-float 0.0) nk0 new-n0))
          (incf (aref z0 k) (- gscale (* (/ (- (sqrt new-n0) (sqrt nk0)) alpha)
                                         (aref bias k))))
          (setf (aref n0 k) new-n0
                (aref bias k)
                (ftrl-weight-of (aref z0 k) new-n0 alpha beta 0.0 lambda2))))))
  learner)

(define-multi-class-learner-train/test-functions softmax+ftrl)

;;; Sparse softmax + FTRL-Proximal
;;;
;;; Identical arithmetic to SOFTMAX+FTRL; only the traversal differs.  The per-class loop
;;; walks the input's index-vector and nothing else, so an update is O(K * nnz).  Z, N and
;;; WEIGHT rows are full-length dense arrays read and written only at the active indices;
;;; untouched coordinates keep z = 0, hence w = 0, which is their initial value.

(defstruct (sparse-softmax+ftrl (:constructor  %make-sparse-softmax+ftrl) (:copier nil)
                                (:print-object %print-sparse-softmax+ftrl))
  input-dimension n-class weight bias
  ;; meta parameters
  alpha beta lambda1 lambda2
  ;; per-coordinate state, the intercept's, and the score/probability scratch
  z n z0 n0 tmp-p)

(defun %print-sparse-softmax+ftrl (obj stream)
  (format stream "#S(SPARSE-SOFTMAX+FTRL~%~T:INPUT-DIMENSION ~A~%~T:N-CLASS ~A~%~T:WEIGHT #(~A ...)~%~T:BIAS ~A~%~T:NONZERO-WEIGHTS ~A/~A)"
          (sparse-softmax+ftrl-input-dimension obj)
          (sparse-softmax+ftrl-n-class obj)
          (%vec-head (svref (sparse-softmax+ftrl-weight obj) 0))
          (%vec-head (sparse-softmax+ftrl-bias obj))
          (reduce #'+ (sparse-softmax+ftrl-weight obj) :initial-value 0
                  :key (lambda (row) (count-if-not #'zerop row)))
          (* (sparse-softmax+ftrl-n-class obj) (sparse-softmax+ftrl-input-dimension obj))))

(defun make-sparse-softmax+ftrl (input-dimension n-class alpha beta lambda1 lambda2)
  (check-type input-dimension integer)
  (check-type n-class integer)
  (check-type alpha number)
  (check-type beta number)
  (check-type lambda1 number)
  (check-type lambda2 number)
  (assert (> input-dimension 0))
  (assert (> n-class 2))
  (let ((alpha (coerce alpha 'single-float))
        (beta (coerce beta 'single-float))
        (lambda1 (coerce lambda1 'single-float))
        (lambda2 (coerce lambda2 'single-float)))
    ;; Assert AFTER coercion; see the comment in MAKE-SOFTMAX+FTRL.
    (assert (< 0.0 alpha))
    (assert (<= 0.0 beta))
    (assert (<= 0.0 lambda1))
    (assert (<= 0.0 lambda2))
    (%make-sparse-softmax+ftrl
     :input-dimension input-dimension
     :n-class n-class
     :weight (%make-weight-vectors n-class input-dimension 0.0)
     :bias (make-vec n-class 0.0)
     :alpha alpha
     :beta beta
     :lambda1 lambda1
     :lambda2 lambda2
     :z (%make-weight-vectors n-class input-dimension 0.0)
     :n (%make-weight-vectors n-class input-dimension 0.0)
     :z0 (make-vec n-class 0.0)
     :n0 (make-vec n-class 0.0)
     :tmp-p (make-vec n-class 0.0))))

(defun sparse-softmax+ftrl-predict (learner input)
  (declare (type sparse-softmax+ftrl learner)
           (type sparse-vector input)
           (optimize (speed 3) (safety 0)))
  (let ((weight (sparse-softmax+ftrl-weight learner))
        (bias (sparse-softmax+ftrl-bias learner))
        (n-class (sparse-softmax+ftrl-n-class learner))
        (max-f most-negative-single-float)
        (max-i 0))
    (declare (type simple-vector weight)
             (type (simple-array single-float) bias)
             (type fixnum n-class max-i)
             (type single-float max-f))
    (loop for k of-type fixnum from 0 below n-class do
      (let ((fk (+ (ds-dot (the (simple-array single-float) (svref weight k)) input)
                   (aref bias k))))
        (declare (type single-float fk))
        (when (> fk max-f)
          (setf max-f fk
                max-i k))))
    max-i))

;; training-label should be an integer class index (0 ... K-1).  The range is NOT
;; checked, for the same reason as SOFTMAX+FTRL-UPDATE above: this body only compares against
;; training-label, never indexes by it, so an out-of-range value causes no fault, only a silently
;; wrong gradient (all K rows pushed down).
(defun sparse-softmax+ftrl-update (learner input training-label)
  (declare (type sparse-softmax+ftrl learner)
           (type sparse-vector input)
           (type fixnum training-label)
           (optimize (speed 3) (safety 0)))
  (let ((weight (sparse-softmax+ftrl-weight learner))
        (bias (sparse-softmax+ftrl-bias learner))
        (z (sparse-softmax+ftrl-z learner))
        (n (sparse-softmax+ftrl-n learner))
        (z0 (sparse-softmax+ftrl-z0 learner))
        (n0 (sparse-softmax+ftrl-n0 learner))
        (p (sparse-softmax+ftrl-tmp-p learner))
        (n-class (sparse-softmax+ftrl-n-class learner))
        (alpha (sparse-softmax+ftrl-alpha learner))
        (beta (sparse-softmax+ftrl-beta learner))
        (lambda1 (sparse-softmax+ftrl-lambda1 learner))
        (lambda2 (sparse-softmax+ftrl-lambda2 learner))
        (index-vector (sparse-vector-index-vector input))
        (value-vector (sparse-vector-value-vector input)))
    (declare (type simple-vector weight z n)
             (type (simple-array single-float) bias z0 n0 p value-vector)
             (type (simple-array fixnum) index-vector)
             (type fixnum n-class)
             (type single-float alpha beta lambda1 lambda2))
    ;; 1. Score every class into P, tracking the maximum.
    (let ((max-f most-negative-single-float))
      (declare (type single-float max-f))
      (loop for k of-type fixnum from 0 below n-class do
        (let ((fk (+ (ds-dot (the (simple-array single-float) (svref weight k)) input)
                     (aref bias k))))
          (declare (type single-float fk))
          (setf (aref p k) fk)
          (when (> fk max-f) (setf max-f fk))))
      ;; 2. Turn P into probabilities.  Subtracting MAX-F before EXP is what keeps this
      ;;    from overflowing.
      (let ((sum 0.0))
        (declare (type single-float sum))
        (loop for k of-type fixnum from 0 below n-class do
          (let ((e (exp (- (aref p k) max-f))))
            (declare (type single-float e))
            (setf (aref p k) e)
            (incf sum e)))
        (loop for k of-type fixnum from 0 below n-class do
          (setf (aref p k) (/ (aref p k) sum)))))
    ;; 3. One FTRL step per class, over the active coordinates only.
    (loop for k of-type fixnum from 0 below n-class do
      (let ((gscale (- (aref p k) (if (= k training-label) 1.0 0.0)))
            (weight-k (svref weight k))
            (z-k (svref z k))
            (n-k (svref n k)))
        (declare (type single-float gscale)
                 (type (simple-array single-float) weight-k z-k n-k))
        (dosvec input j
          (let* ((i (aref index-vector j))
                 (gi (* gscale (aref value-vector j)))
                 (ni (aref n-k i))
                 (new-ni (+ ni (* gi gi))))
            (declare (type fixnum i)
                     (type single-float gi)
                     (type (single-float 0.0) ni new-ni))
            (incf (aref z-k i) (- gi (* (/ (- (sqrt new-ni) (sqrt ni)) alpha)
                                        (aref weight-k i))))
            (setf (aref n-k i) new-ni
                  (aref weight-k i)
                  (ftrl-weight-of (aref z-k i) new-ni alpha beta lambda1 lambda2))))
        ;; The intercept is an always-1 feature and is deliberately NOT L1-regularised.
        (let* ((nk0 (aref n0 k))
               (new-n0 (+ nk0 (* gscale gscale))))
          (declare (type (single-float 0.0) nk0 new-n0))
          (incf (aref z0 k) (- gscale (* (/ (- (sqrt new-n0) (sqrt nk0)) alpha)
                                         (aref bias k))))
          (setf (aref n0 k) new-n0
                (aref bias k)
                (ftrl-weight-of (aref z0 k) new-n0 alpha beta 0.0 lambda2))))))
  learner)

(define-multi-class-learner-train/test-functions sparse-softmax+ftrl)

;;; Deep copies of learners
;;;
;;; DEFSTRUCT's own copier is shallow and every learner here keeps its state in arrays, so
;;; a shallow copy shares exactly the thing a caller wanted to snapshot: training the
;;; original rewrites the copy. cl-random-forest's TRAIN-REFINE-LEARNER-PROCESS was doing
;;; that -- snapshotting before each epoch to roll back to the best one, and always getting
;;; the last one instead (masatoi/cl-random-forest#20). Every learner struct is therefore
;;; declared (:copier nil) and given a deep COPY-<TYPE> under the name DEFSTRUCT would have
;;; used, so the shallow version is not reachable by accident.
;;;
;;; Which slots hold state is decided at run time rather than listed per struct. Passing a
;;; scalar, a symbol or a function through %COPY-LEARNER-STATE returns it unchanged, so a
;;; copier names every slot of its struct and cannot misclassify one -- the failure mode
;;; being guarded against is a slot quietly left shared, and requiring the author to sort
;;; slots into two lists is how that happens.

(defgeneric copy-learner (learner)
  (:documentation
   "Return an independent deep copy of LEARNER, whatever kind of learner it is.

Training the copy does not touch the original and training the original does not touch the
copy, which is what makes it usable as a snapshot. A learner type with no method signals
NO-APPLICABLE-METHOD rather than falling back on a shallow copy -- finding out by way of a
snapshot that silently tracked its original is the failure this exists to prevent."))

(defun %copy-learner-state (value)
  "Duplicate VALUE if it is mutable learner state, otherwise return it unchanged.

Mutable state is a specialised float vector, or a SIMPLE-VECTOR of them -- the shape
%MAKE-WEIGHT-VECTORS produces for the multiclass learners. Arrays of any other shape and
nested structures signal, rather than being shared: a new slot shape must be handled here
deliberately, because sharing one is silent and only shows up as a snapshot that tracks
its original."
  (typecase value
    ((simple-array single-float (*)) (copy-seq value))
    (simple-vector (map 'simple-vector #'%copy-learner-state value))
    (structure-object
     (error "%COPY-LEARNER-STATE reached a ~S. Nested learners need an explicit copier ~
that recurses through COPY-LEARNER, as COPY-ONE-VS-REST does."
            (type-of value)))
    (array
     (error "%COPY-LEARNER-STATE cannot duplicate a ~S. Extend it before giving a learner ~
a slot of that shape." (type-of value)))
    (t value)))

(defmacro define-learner-copier (learner-type &rest slots)
  "Define COPY-<LEARNER-TYPE> as a deep copy of a learner struct.

SLOTS should name every slot the struct has. Naming a scalar slot is harmless -- it goes
through %COPY-LEARNER-STATE unchanged -- and that is deliberate: it means the safe thing
to do is list them all."
  (let ((copier (intern (catstr "COPY-" (symbol-name learner-type))))
        (new (gensym "COPY")))
    `(progn
       (defun ,copier (learner)
         ,(format nil "Return a deep copy of LEARNER, sharing no ~A array with it."
                  (string-downcase (symbol-name learner-type)))
         (declare (type ,learner-type learner))
         (let ((,new (copy-structure learner)))
           (setf ,@(loop for slot in slots
                         for accessor = (intern (catstr (symbol-name learner-type)
                                                        (catstr "-" (symbol-name slot))))
                         append `((,accessor ,new) (%copy-learner-state (,accessor ,new)))))
           ,new))
       (defmethod copy-learner ((learner ,learner-type))
         (,copier learner)))))

(define-learner-copier perceptron
  input-dimension weight bias tmp-float)

(define-learner-copier arow
  input-dimension weight bias gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-float)

(define-learner-copier scw
  input-dimension weight bias eta C phi psi zeta sigma sigma0
  tmp-vec1 tmp-vec2 tmp-float)

(define-learner-copier lr+sgd
  input-dimension weight bias C eta g tmp-vec tmp-float)

(define-learner-copier lr+adam
  input-dimension weight bias C alpha epsilon beta1 beta2
  g m v m0 v0 beta1^t beta2^t tmp-vec tmp-float)

(define-learner-copier lr+ftrl
  input-dimension weight bias alpha beta lambda1 lambda2 z n z0 n0)

(define-learner-copier sparse-perceptron
  input-dimension weight bias tmp-float)

(define-learner-copier sparse-arow
  input-dimension weight bias gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-float)

(define-learner-copier sparse-scw
  input-dimension weight bias eta C phi psi zeta sigma sigma0
  tmp-vec1 tmp-vec2 tmp-float)

(define-learner-copier sparse-lr+sgd
  input-dimension weight bias C eta g tmp-vec tmp-float)

(define-learner-copier sparse-lr+adam
  input-dimension weight bias C alpha epsilon beta1 beta2
  g m v m0 v0 beta1^t beta2^t tmp-vec tmp-float)

(define-learner-copier sparse-lr+ftrl
  input-dimension weight bias alpha beta lambda1 lambda2 z n z0 n0)

(define-learner-copier multiclass-arow
  input-dimension n-class weight bias gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-vec3)

(define-learner-copier sparse-multiclass-arow
  input-dimension n-class weight bias gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-vec3)

(define-learner-copier softmax+ftrl
  input-dimension n-class weight bias alpha beta lambda1 lambda2 z n z0 n0 tmp-p)

(define-learner-copier sparse-softmax+ftrl
  input-dimension n-class weight bias alpha beta lambda1 lambda2 z n z0 n0 tmp-p)

(defun copy-one-vs-rest (learner)
  "Return a deep copy of LEARNER, including every sub-learner.

The four function slots are shared rather than copied. They are stateless dispatch
closures determined by the sub-learner type -- the same ones
ONE-VS-REST-RESTORE-FUNCTIONS rebuilds from scratch after a CL-STORE round trip."
  (declare (type one-vs-rest learner))
  (let ((new (copy-structure learner)))
    (setf (one-vs-rest-learners-vector new)
          (map 'simple-vector #'copy-learner (one-vs-rest-learners-vector new)))
    new))

(defun copy-one-vs-one (learner)
  "Return a deep copy of LEARNER, including every sub-learner.
The two function slots are shared, for the reason COPY-ONE-VS-REST gives."
  (declare (type one-vs-one learner))
  (let ((new (copy-structure learner)))
    (setf (one-vs-one-learners-vector new)
          (map 'simple-vector #'copy-learner (one-vs-one-learners-vector new)))
    new))

(defmethod copy-learner ((learner one-vs-rest))
  (copy-one-vs-rest learner))

(defmethod copy-learner ((learner one-vs-one))
  (copy-one-vs-one learner))

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
