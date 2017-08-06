;;; -*- coding:utf-8; mode: lisp; -*-

(in-package :cl-user)
(defpackage cl-online-learning-asd
  (:use :cl :asdf))
(in-package :cl-online-learning-asd)

(defsystem cl-online-learning
  :version "0.4"
  :author "Satoshi Imai"
  :licence "MIT Licence"
  :encoding :utf-8
  :depends-on (:split-sequence :parse-number)
  :components ((:module "src"
			:components
			((:file "vector")
                         (:file "utils" :depends-on ("vector"))
                         (:file "cl-online-learning" :depends-on ("vector"))
                         (:file "rls" :depends-on ("vector" "cl-online-learning")))
                        ))
  :description "Online Machine Learning for Common Lisp"
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.org"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (test-op cl-online-learning-test))))
