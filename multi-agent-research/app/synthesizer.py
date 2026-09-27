from app.contracts import ResearchReport, ResearchResult


class ResearchSynthesizer:
    def __init__(self, provider):
        self.provider = provider

    def synthesize(
        self,
        request: str,
        results: list[ResearchResult],
    ) -> ResearchReport:
        if not isinstance(request, str):
            raise ValueError("Request must be a string")

        if not request.strip():
            raise ValueError("Request cannot be empty")

        if not results:
            raise ValueError("At least one research result is required")

        if not all(isinstance(result, ResearchResult) for result in results):
            raise ValueError("Results must contain ResearchResult objects")

        synthesis = self.provider.synthesize(
            request=request,
            results=results,
        )

        return ResearchReport(
            request=request,
            results=tuple(results),
            synthesis=synthesis,
        )
