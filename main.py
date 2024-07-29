@app.route('/api/v1.0/students', methods=['POST'])
def students():
    try:
        # Read the CSV content from the request body
        csv_content = request.data.decode('utf-8')
        
        # Use StringIO to simulate a file object for pandas
        csv_file = io.StringIO(csv_content)
        
        # Attempt to load the CSV content into a DataFrame
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.ParserError as e:
            return jsonify({'error': f'CSV parsing error: {str(e)}'}), 400
        
        # Check if the DataFrame is empty
        if df.empty:
            return jsonify({'error': 'Empty CSV data'}), 400
        
        # Ensure expected columns are present
        required_columns = {'company name', 'description'}
        if not required_columns.issubset(df.columns):
            return jsonify({'error': f'Missing columns in CSV. Required columns are: {required_columns}'}), 400
        
        # Define the list of questions
        questions = [
            "What does {} provide?",
            "What is {} vertical focus?",
            "Who are {} consumers?"
        ]
        
        # Initialize a list to store responses
        responses = []
        
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            description = row.get('description', '')
            company_name = row.get('company name', '')
            
            # Prepare responses for the current row
            row_responses = []
            for question in questions:
                formatted_question = question.format(company_name)
                
                # Prepare the QA input for the pipeline
                QA_input = {
                    'question': formatted_question,
                    'context': description
                }
                
                # Get the answer from the pipeline
                answer = qa_pipeline(QA_input)['answer']
                row_responses.append(answer)
            
            # Append the row responses to the list of responses
            responses.append(row_responses)
        
        # Convert responses to a DataFrame
        responses_df = pd.DataFrame(responses, columns=['Question 1', 'Question 2', 'Question 3'])
        
        # Split into individual columns for each question
        responses_question1 = responses_df['Question 1'].tolist()
        responses_question2 = responses_df['Question 2'].tolist()
        responses_question3 = responses_df['Question 3'].tolist()
        
        # Initialize similarity arrays
        similarity_question1 = []
        similarity_question2 = []
        similarity_question3 = []
        
        # Compute similarities for responses_question1
        for i in range(len(responses_question1)):
            text1 = responses_question1[0]
            text2 = responses_question1[i]
            embedding1 = sbert_model.encode(text1, convert_to_tensor=True)
            embedding2 = sbert_model.encode(text2, convert_to_tensor=True)
            cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            similarity_question1.append(cosine_similarity)
        
        # Compute similarities for responses_question2
        for i in range(len(responses_question2)):
            text1 = responses_question2[0]
            text2 = responses_question2[i]
            embedding1 = sbert_model.encode(text1, convert_to_tensor=True)
            embedding2 = sbert_model.encode(text2, convert_to_tensor=True)
            cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            similarity_question2.append(cosine_similarity)
        
        # Compute similarities for responses_question3
        for i in range(len(responses_question3)):
            text1 = responses_question3[0]
            text2 = responses_question3[i]
            embedding1 = sbert_model.encode(text1, convert_to_tensor=True)
            embedding2 = sbert_model.encode(text2, convert_to_tensor=True)
            cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            similarity_question3.append(cosine_similarity)
        
        # Combine similarities into a DataFrame
        similarity_df = pd.DataFrame({
            'similarity_question1': similarity_question1,
            'similarity_question2': similarity_question2,
            'similarity_question3': similarity_question3
        })
        
        # Convert DataFrame to the appropriate format for model prediction
        features = similarity_df.values
        
        # Make predictions
        predictions = regression_model.predict(features)
        
        # Prepare the result
        result = {
            'responses': responses,
            'similarities': similarity_df.to_dict(orient='records'),
            'predictions': predictions.tolist()
        }
        
        # Return the result as JSON
        return jsonify(result)

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500
