import Foundation

/// Пользовательский словарь замен: правит устойчивые ошибки распознавания
/// («ватзап» → «Wazzup») до любой обработки и вставки текста.
///
/// Правила лежат в обычном текстовом файле
/// `~/Library/Application Support/VoiceSwitch/replacements.txt` — по одному
/// на строку, «что → чем» (годится и «->»). Файл перечитывается на каждую
/// расшифровку, поэтому правки применяются без перезапуска приложения.
enum Replacements {
    private struct Rule {
        let pattern: NSRegularExpression
        let replacement: String
    }

    static var fileURL: URL {
        RuntimePaths.applicationSupportRoot.appendingPathComponent("replacements.txt")
    }

    private static let template = """
    # Словарь замен VoiceSwitch: по правилу на строку, «что → чем» (можно «->»).
    # Сравнение без учёта регистра, по целым словам; словоформы («в ватзапе»)
    # добавляйте отдельными строками. Строки с # и пустые пропускаются.
    # Пример:
    # ватзап → Wazzup
    """

    static func apply(to text: String) -> String {
        var result = text
        for rule in loadRules() {
            result = rule.pattern.stringByReplacingMatches(
                in: result,
                range: NSRange(result.startIndex..., in: result),
                withTemplate: rule.replacement
            )
        }
        return result
    }

    private static func loadRules() -> [Rule] {
        guard let raw = try? String(contentsOf: fileURL, encoding: .utf8) else {
            try? FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? template.write(to: fileURL, atomically: true, encoding: .utf8)
            return []
        }
        return raw.split(separator: "\n").compactMap { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { return nil }
            let parts = trimmed.components(separatedBy: "→")
                .flatMap { $0.components(separatedBy: "->") }
            guard parts.count == 2 else { return nil }
            let source = parts[0].trimmingCharacters(in: .whitespaces)
            let target = parts[1].trimmingCharacters(in: .whitespaces)
            guard !source.isEmpty else { return nil }
            // Целое слово: слева и справа не должно быть буквы или цифры.
            guard let pattern = try? NSRegularExpression(
                pattern: "(?<![\\p{L}\\p{N}])"
                    + NSRegularExpression.escapedPattern(for: source)
                    + "(?![\\p{L}\\p{N}])",
                options: [.caseInsensitive]
            ) else { return nil }
            return Rule(
                pattern: pattern,
                replacement: NSRegularExpression.escapedTemplate(for: target)
            )
        }
    }
}

extension TranscriptionResult {
    func applyingReplacements() -> TranscriptionResult {
        TranscriptionResult(
            requestID: requestID,
            engine: engine,
            text: Replacements.apply(to: text),
            latency: latency,
            audioDuration: audioDuration,
            detectedLanguage: detectedLanguage
        )
    }
}
