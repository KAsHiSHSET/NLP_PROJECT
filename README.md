#Hindi-Question-Answering-System

Question Answering(QA) is a type of information extraction system which provides specific answers for the questions being asked. When we search for information on the Internet, we employ information retrieval systems which provide the entire web page as a response. However, as they show you documents, you must read them and decide whether they include the information you desire. A question answering system will make it possible to find the information effectively. These systems are already implemented in a variety of languages, but not very much in Indian languages. The proposed system is a Question Answer System (QA) for Hindi material that would be useful to persons who learn Hindi as their first language. The Hindi text is evaluated to determine the semantics of each line, and the appropriate answer for the given question is retrieved from the document using span texts; the model was able to accurately extract answers from passages.

#Steps involves in buiding the system
#Pre-Processing
In the preprocessing of passages, they are broken down into smaller units (individual terms or a set of words), which are called tokens. This is significant because the text's meaning may be easily deduced by examining the words in the text. The answers are processed by marking all the character indexes in the passage that are also in the answer then finding all the tokens that are in the answers and finally storing start and end token indexes. These start and end tokens will help the model understand the connection between a question and its answer.

#Span Texts
After the preprocessing of passages is done, span texts are generated using the preprocessed passages. Span texts are used to understand the semantic meaning of the sentence. The whole passage after tokenization is divided into span texts of 128 words, out of which only one set will have the correct answer, this way the model will understand where to look for the answer in the passage.

hindi text(around 200 words) = “कुछ नॉर्मनर्म सदुरू पर्वीू र्वी अनातोलि या मेंअर्मेनि याई के …… जबकि अमाल्फ़ि और बारी इटली मेंनॉर्मनर्म शासन के अधीन थे।”

span texts[question, span_context, span_answer_start, answer_text, span_answer_exist]: This is the format in which the span texts are stored.

Question: It is the question being asked.

span_context: It contains the actual span text consisting of 128 words or tokens.

span_answer_start: It indicates the starting position of answer in the span text.

answer_text: It is the answer for the question being asked.

span_answer_exist: It indicates whether the answer exists in the span text or not.

#Examples:

['अनटोलि या मेंनॉर्मन्र्म स कि सकी टीम मेंशामि ल थे?', "कुछ नॉर्मनर्म सदुरू पर्वीू र्वी अनातोलि या मेंअर्मेनि याई ……बीच का ज्ञात व्यापार उन शहरों मेंइटालो-नॉर्मन्र्म स की", 114, 'तर्कीु र्की सेना', 1],

['अनटोलि या मेंनॉर्मन्र्म स कि सकी टीम मेंशामि ल थे?', "से1074 के मध्य अर्मेनि याई जनरल …….. बारी इटली मेंनॉर्मनर्म शासन के अधीन थे।", 0, 'तर्कीु र्की सेना', 0]

As a result of making span texts, the model will understand where to look for the answers in the passage. It can capture the semantic meaning of texts surrounding the answer in a shorter sentence as compared to a larger sentence.

#Pre-Processing of questions: The preprocessing of questions is
done in a similar way as is done for the passages. Questions are also tokenized to capture their semantic meaning, this will help the model to understand the meaning of the question being asked and where to find the answer in the passage.

#Model:
The BERT model is used to train the data because it can look at the words that follow before and after a word to get its whole context, which is particularly valuable for determining the intent behind the question asked. The traditional use of BERT for QA involves packing both questions and the reference text and giving it as an input to the model; the 2 pieces are separated by a [sep] token. Start and end token classifier is used to predict the tokens in the answer. This approach has been applied to English language and a few other foreign languages.

The proposed work works in a similar way with the addition of span texts as an input to the model. Input IDs, token type IDs, and attention masks are the model's inputs. Input IDs are self-explanatory: they're just mappings between tokens and their corresponding IDs. The model's attention mask prevents it from looking at padding tokens. Token type IDs are often used in next word prediction tasks, in which two sentences are provided; in the proposed work, it is used for predicting tokens in the answer. The output of the model is the probability of the starting and the ending token of the answer which is present in the passage. The probability is produced by applying the softmax function to the dot product of start token weights and word embeddings.

#Answer Prediction:
Once the model has predicted the start and end tokens for the answer, it will use these tokens to map the corresponding text in the passage and hence the answer will be detected.

#Implementation
The model is trained on the xquad dataset which contains around 240 unique passages and around 1200 unique questions covering different segments of literature like sports, culture politics, history, etc, in the hindi language. The model makes use of pre-trained text embeddings along with the embeddings from passages, questions and answers. The dataset contains around 1200 questions and the whole dataset was split into 80:20 ratio as training and testing set .The loss function used is Sparse categorical cross entropy because it saves time and memory as well as computation, as it simply uses a single integer for a class. The model is trained on a GPU for 20 epochs minimizing the loss value and improving the accuracy of the model. Adam optimizer is used as it is good at handling noisy and sparse gradients.

Results of model on training and testing set

![image](https://github.com/user-attachments/assets/a217f8a6-ab47-4e21-aea5-b2cb6cad6ca4)

UI Design


![image](https://github.com/user-attachments/assets/fa2d3280-0887-49f1-a183-5a4b0b7712c4)

![image](https://github.com/user-attachments/assets/80e063f6-db2c-42ed-9e4c-2a4de7c1eeb0)

