;;; -*- coding:utf-8; mode: lisp; -*-

(in-package :cl-user)
(defpackage cl-online-learning-asd
  (:use :cl :asdf))
(in-package :cl-online-learning-asd)

(defsystem cl-online-learning
  :version "0.1"
  :author "Satoshi Imai"
  :licence "MIT Licence"
  :encoding :utf-8
  :depends-on (:cl-ppcre :parse-number)
  :components ((:module "src"
			:components
			((:file "utils")
			 (:file "vector" :depends-on ("utils"))
			 (:file "cl-online-learning" :depends-on ("utils" "vector"))
			 (:file "cl-online-learning-utils"))))
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
