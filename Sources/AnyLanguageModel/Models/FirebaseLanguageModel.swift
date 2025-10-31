// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
#if FirebaseAILogic
    import FirebaseAILogic

    public struct FirebaseLanguageModel: LanguageModel {
        let firebaseAI: FirebaseAI
        let modelName: String

        public init(firebaseAI: FirebaseAI, modelName: String) {
            self.firebaseAI = firebaseAI
            self.modelName = modelName
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            let model = firebaseAI.generativeModel(
                modelName: modelName,
                generationConfig: try options.asGenerationConfig(generating: type),
                systemInstruction: session.instructions?.asSystemInstruction
            )

            let response = try await model.generateContent(prompt.description)

            if type == String.self, let text = response.text {
                let generatedContent = GeneratedContent(kind: .string(text))
                return .init(
                    content: try Content(generatedContent),
                    rawContent: generatedContent,
                    transcriptEntries: ArraySlice()
                )
            } else if let json = response.text {
                let generatedContent = try GeneratedContent(json: json)
                return .init(
                    content: try Content(generatedContent),
                    rawContent: generatedContent,
                    transcriptEntries: ArraySlice()
                )
            } else {
                fatalError()
            }
        }

        public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            fatalError()
        }
    }

    // MARK: - Private Helpers

    private extension GenerationSchema {
        func asJSONSchema() throws -> JSONObject {
            let jsonEncoder = JSONEncoder()
            let jsonData = try jsonEncoder.encode(self)
            let jsonDecoder = JSONDecoder()
            var jsonSchema = try jsonDecoder.decode(JSONObject.self, from: jsonData)
            if let orderList = jsonSchema.removeValue(forKey: "x-order") {
                jsonSchema["propertyOrdering"] = orderList
            }
            return jsonSchema
        }
    }

    private extension Instructions {
        public var asSystemInstruction: ModelContent {
            // TODO: Find a better way to get the textual representation of the Instructions.
            return ModelContent(role: "system", parts: self.description)
        }
    }

    private extension GenerationOptions {
        func asGenerationConfig<Content>(
            generating type: Content.Type
        ) throws -> GenerationConfig where Content: Generable {
            let responseMIMEType: String?
            let responseJSONSchema: JSONObject?
            if type == String.self {
                responseMIMEType = nil
                responseJSONSchema = nil
            } else {
                responseMIMEType = "application/json"
                responseJSONSchema = try type.generationSchema.asJSONSchema()
            }

            let topP: Float?
            let topK: Int?
            if case let .nucleus(probabilityThreshold, _) = self.sampling?.mode {
                topP = Float(probabilityThreshold)
                topK = nil
            } else if case let .topK(topKValue, _) = self.sampling?.mode {
                topP = nil
                topK = topKValue
            } else if case .greedy = self.sampling?.mode {
                topP = 0
                topK = 0
            } else {
                topP = nil
                topK = nil
            }

            let temperature: Float?
            if let optionsTemperature = self.temperature {
                temperature = Float(optionsTemperature)
            } else {
                temperature = nil
            }

            if let responseMIMEType, let responseJSONSchema {
                return GenerationConfig(
                    temperature: temperature,
                    topP: topP,
                    topK: topK,
                    maxOutputTokens: self.maximumResponseTokens,
                    responseMIMEType: responseMIMEType,
                    responseJSONSchema: responseJSONSchema
                )
            } else {
                return GenerationConfig(
                    temperature: temperature,
                    topP: topP,
                    topK: topK,
                    maxOutputTokens: self.maximumResponseTokens
                )
            }
        }
    }
#endif  // FirebaseAILogic
