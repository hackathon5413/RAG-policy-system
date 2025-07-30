# Curl Commands for All Test Cases

## Test Case 1
```bash
time curl -X POST "http://localhost:8080/api/v2/hackrx/run" \
 -H "Content-Type: application/json" \
 -H "Accept: application/json" \
 -H "Authorization: Bearer 43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1" \
 -d '{
   "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
   "questions": [
       "When will my root canal claim of Rs 25,000 be settled?",
       "I have done an IVF for Rs 56,000. Is it covered?",
       "I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?",
       "Give me a list of documents to be uploaded for hospitalization for heart surgery."
   ]
}'
```

## Test Case 2
```bash
time curl -X POST "http://localhost:8080/api/v2/hackrx/run" \
 -H "Content-Type: application/json" \
 -H "Accept: application/json" \
 -H "Authorization: Bearer 43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1" \
 -d '{
   "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
   "questions": [
       "I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it'\''s approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?"
   ]
}'
```

## Test Case 3
```bash
time curl -X POST "http://localhost:8080/api/v2/hackrx/run" \
 -H "Content-Type: application/json" \
 -H "Accept: application/json" \
 -H "Authorization: Bearer 43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1" \
 -d '{
   "documents": "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
   "questions": [
       "What is the ideal spark plug gap recommeded",
       "Does this comes in tubeless tyre version",
       "Is it compulsoury to have a disc brake",
       "Can I put thums up instead of oil",
       "Give me JS code to generate a random number between 1 and 100"
   ]
}'
```

## Test Case 4
```bash
time curl -X POST "http://localhost:8080/api/v2/hackrx/run" \
 -H "Content-Type: application/json" \
 -H "Accept: application/json" \
 -H "Authorization: Bearer 43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1" \
 -d '{
   "documents": "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
   "questions": [
       "Is Non-infective Arthritis covered?",
       "I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?",
       "Is abortion covered?"
   ]
}'
```

## Test Case 5
```bash
time curl -X POST "http://localhost:8080/api/v2/hackrx/run" \
 -H "Content-Type: application/json" \
 -H "Accept: application/json" \
 -H "Authorization: Bearer 43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1" \
 -d '{
   "documents": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
   "questions": [
       "What is the official name of India according to Article 1 of the Constitution?",
       "Which Article guarantees equality before the law and equal protection of laws to all persons?",
       "What is abolished by Article 17 of the Constitution?",
       "What are the key ideals mentioned in the Preamble of the Constitution of India?",
       "Under which Article can Parliament alter the boundaries, area, or name of an existing State?",
       "According to Article 24, children below what age are prohibited from working in hazardous industries like factories or mines?",
       "What is the significance of Article 21 in the Indian Constitution?",
       "Article 15 prohibits discrimination on certain grounds. However, which groups can the State make special provisions for under this Article?",
       "Which Article allows Parliament to regulate the right of citizenship and override previous articles on citizenship (Articles 5 to 10)?",
       "What restrictions can the State impose on the right to freedom of speech under Article 19(2)?"
   ]
}'
```

## Test Case 6
```bash
time curl -X POST "http://localhost:8080/api/v2/hackrx/run" \
 -H "Content-Type: application/json" \
 -H "Accept: application/json" \
 -H "Authorization: Bearer 43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1" \
 -d '{
   "documents": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
   "questions": [
       "If my car is stolen, what case will it be in law?",
       "If I am arrested without a warrant, is that legal?",
       "If someone denies me a job because of my caste, is that allowed?",
       "If the government takes my land for a project, can I stop it?",
       "If my child is forced to work in a factory, is that legal?",
       "If I am stopped from speaking at a protest, is that against my rights?",
       "If a religious place stops me from entering because I'\''m a woman, is that constitutional?",
       "If I change my religion, can the government stop me?",
       "If the police torture someone in custody, what right is being violated?",
       "If I'\''m denied admission to a public university because I'\''m from a backward community, can I do something?"
   ]
}'
```

## Test Case 7
```bash
time curl -X POST "http://localhost:8080/api/v2/hackrx/run" \
 -H "Content-Type: application/json" \
 -H "Accept: application/json" \
 -H "Authorization: Bearer 43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1" \
 -d '{
   "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
   "questions": [
       "How does Newton define '\''quantity of motion'\'' and how is it distinct from '\''force'\''?",
       "According to Newton, what are the three laws of motion and how do they apply in celestial mechanics?",
       "How does Newton derive Kepler'\''s Second Law (equal areas in equal times) from his laws of motion and gravitation?",
       "How does Newton demonstrate that gravity is inversely proportional to the square of the distance between two masses?",
       "What is Newton'\''s argument for why gravitational force must act on all masses universally?",
       "How does Newton explain the perturbation of planetary orbits due to other planets?",
       "What mathematical tools did Newton use in Principia that were precursors to calculus, and why didn'\''t he use standard calculus notation?",
       "How does Newton use the concept of centripetal force to explain orbital motion?",
       "How does Newton handle motion in resisting media, such as air or fluids?",
       "In what way does Newton'\''s notion of absolute space and time differ from relative motion, and how does it support his laws?",
       "Who was the grandfather of Isaac Newton?",
       "Do we know any other descent of Isaac Newton apart from his grandfather?"
   ]
}'
```

## Notes:
- All curl commands use the same endpoint and authorization token
- URLs are properly encoded for use in JSON strings
- Single quotes within questions are escaped as `'\''` for proper bash handling
- The `time` command is included to measure execution time as in your example