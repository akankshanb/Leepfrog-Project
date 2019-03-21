# Leepfrog-Project
# Description:
Here at CourseLeaf we are interested in enchancing student experience. One way we can do this is by helping students, who have yet to declare a major, by recommending a list of majors they might be interested in.
We think that if we have historical data of student course records and their majors, we should be able to predict potential majors for new students who have yet to declare their majors.
# Problem Statement:
Given the historic student records, we would like you to create a machine learning model which is able to predict the majors of new students.
Attached are 2 pipe-seperated files with student data: training.psv and eval.psv.
training.psv has the historical data of about 10,000 students who have declared their majors while eval.psv has the data of about 2,000 students who have yet to declare their majors.
The data in training.psv is of the form: student_id|level|course|grade|major
Where student_id is the anonymized unique identifier of a student.
The level is whether they were a freshman, sophomore, junior or senior when they enrolled in a course. The course is the name of the course and the grade is their final grade in that course.
The major is their final declared major.
The possible grades a student can get in a course are:
     <b> Grade </b> </br>
     A+ </br>
     A </br>
     A- </br>
     B+ </br>
     B </br>
     B- </br>
     C+ </br>
     C </br>
     C- </br>
     D+ </br>
     D </br>
     D- </br>
     F </br>
     AUS (Audit Successful) </br>
     AUU (Audit Unsuccessful) </br>
     IP (In Progress) </br>
     N (Nonpass) </br>
     P (Pass) </br>
     S (Satisfactory) </br>
     U (Unsatisfactory) </br>
 
As an example, the following snippet of data can be interpreted to mean: </br>
EFPdweLY5jycYAVg|Freshman|THTR:114|B|Business </br> EFPdweLY5jycYAVg|Freshman|ASIA:260|D-|Business </br> EFPdweLY5jycYAVg|Junior|CSI:180|C|Business </br> EFPdweLY5jycYAVg|Freshman|ECON:110|C-|Business </br>
A student with the id “EFPdweLY5jycYAVg” enrolled in THTR:114, ASIA:260, ECON:110 in their freshman year and got a B,D-,C- in those courses respectively. The student also enrolled in CSI:180 in their Junior year and passed with a “C”. At the end of their four years, the student graduated with a major in Business. </br>
Similarly, the data in eval.psv is of the form: student_id|level|course|grade|major1|major2|major3 </br>
However, the entries in the major1|major2|major3 columns are missing. 
We expect that you would use the data in training.psv to create a major prediction model. And use that
model to predict majors for the students in eval.psv. </br>
<b> Hint: </b> </br>
1. You can split the data in training.psv into a training and a testing set to evaluate the accuracy of your model.
2. The data can have noise.
Submission:
For each entry in the eval.psv dataset you may predict up to 3 majors (they can be in any order). The
final file should have a header and have the following format:
student_id|level|course|grade|major1|major2|major3 NcBGR8zst63tyLwa|Sophomore|SOC:102|A|Criminology|Business|Sociology NcBGR8zst63tyLwa|Freshman|SPAN:100|C+|Criminology|Business|Sociology </br>
You will be submitting this update eval.psv with the predictions along with a small write up which explains your methodology, including how the data was preprocessed, which algorithms you used to generate your model, and the rationale behind your choice to do so. We will not be accepting any questions. If you have questions, please make an assumption and explain any assumptions that you have made in the write up.
 
