from __future__ import annotations

import json
from typing import Any, Callable

from backend.artifacts.filesystem import ArtifactRun
from backend.pipelines.authorship import add_author_credit
from backend.pipelines.common import (
    PipelineError,
    PipelineResult,
    extract_fenced_or_raw,
    format_citations,
    format_log,
)
from backend.pipelines.html_to_pdf import (
    A4_PAGE,
    ConvertError,
    HtmlToPdfConverter,
)
from backend.providers.base import (
    ModelProviderError,
    TextGenerationProvider,
    TextGenerationRequest,
)
from backend.timing import RunTimingRecorder, measure_stage


def run_essay_html_pipeline(
    *,
    artifact_run: ArtifactRun,
    model_profile: dict[str, Any],
    model_provider: TextGenerationProvider,
    pdf_converter: HtmlToPdfConverter,
    task_text: str,
    context_bundle: str,
    output_preference: str | None,
    options: dict[str, Any] | None,
    search: dict[str, Any],
    max_output_tokens: int,
    emit_event: Callable[[str, str], None] | None = None,
    timing: RunTimingRecorder | None = None,
) -> PipelineResult:
    log_lines = [
        "Pipeline: essay_html",
        f"Output preference: {output_preference or 'pdf'}",
        "Stage: generate_source",
    ]
    system_prompt, user_prompt = build_essay_html_prompt(
        task_text=task_text,
        context_bundle=context_bundle,
        options=options or {},
        search=search,
    )
    if emit_event:
        emit_event("generate_source", "Generating essay HTML source")

    try:
        with measure_stage(timing, "provider_generation"):
            raw_output = model_provider.generate_text(
                TextGenerationRequest(
                    profile=model_profile,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=0.2,
                )
            )
    except ModelProviderError as exc:
        raise PipelineError(
            exc.code,
            exc.message,
            stage="generate_source",
            log_lines=log_lines,
        ) from exc
    except Exception as exc:
        raise PipelineError(
            "provider_unavailable",
            "The model provider request failed.",
            stage="generate_source",
            log_lines=log_lines,
        ) from exc

    source = extract_fenced_or_raw(raw_output, accepted_languages={"html", "htm"})
    source = add_author_credit(source.strip()) + "\n"
    html_path = artifact_run.write_output(
        "main.html",
        source,
        kind="source",
        media_type="text/html",
    )
    log_lines.extend(["Output: output/main.html", "Stage: convert_pdf"])
    if emit_event:
        emit_event("convert_pdf", "Converting essay HTML to PDF")

    pdf_path = artifact_run.run_dir / "output" / "main.pdf"
    try:
        with measure_stage(timing, "convert_pdf"):
            convert_result = pdf_converter.convert(
                html_path=html_path,
                pdf_path=pdf_path,
                page_config=A4_PAGE,
            )
    except ConvertError as exc:
        artifact_run.write_log("convert.log", exc.log_text)
        raise PipelineError(
            "convert_failed",
            exc.message,
            stage="convert_pdf",
            log_lines=log_lines,
        ) from exc
    except Exception as exc:
        artifact_run.write_log(
            "convert.log",
            "HTML-to-PDF converter raised an unexpected error.\n",
        )
        raise PipelineError(
            "convert_failed",
            "HTML-to-PDF conversion failed.",
            stage="convert_pdf",
            log_lines=log_lines,
        ) from exc

    artifact_run.write_log("convert.log", convert_result.log_text)
    artifact_run.write_output(
        "main.pdf",
        convert_result.pdf_path.read_bytes(),
        kind="pdf",
        media_type="application/pdf",
    )
    log_lines.extend(["Convert: pdf_ok", "Output: output/main.pdf"])
    return PipelineResult(log_text=format_log(log_lines))


def build_essay_html_prompt(
    *,
    task_text: str,
    context_bundle: str,
    options: dict[str, Any],
    search: dict[str, Any],
) -> tuple[str, str]:

    system_prompt = (
        "You are an expert academic writing assistant. Your task is to "
        "produce accurate, well-structured, evidence-based academic reports "
        "as complete, self-contained HTML documents. Follow the assignment "
        "requirements closely, prioritize the supplied context and sources, "
        "and never invent facts, quotations, references, or research findings. "
        "Return only the requested HTML source, with no surrounding explanation."
    )

    output_instruction = (
        "Return one complete, self-contained HTML document. It must start "
        "with <!doctype html> and include <html>, <head>, and <body>. "
        "Use semantic HTML5 elements and a meaningful document title. "
        "Organize the report according to the assignment requirements, with "
        "a clear title, introduction, logically ordered sections, analysis, "
        "conclusion, and references when appropriate. Do not force irrelevant "
        "sections if the assignment requires a different structure. "
        "Address every explicit requirement and evaluation criterion in the "
        "assignment. Support important claims with supplied evidence, and "
        "distinguish established facts from assumptions or limitations. "
        "Use only sources actually supplied or verified in the provided "
        "context. Never fabricate citations, URLs, quotations, or references. "
        "If evidence is insufficient, state the limitation honestly. "
        "Write in the language requested by the assignment and use clear, "
        "formal academic prose. Avoid repetition and unsupported claims. "
        "Include all CSS inside a <style> element. Use a readable serif "
        "academic print layout, consistent typography, clear heading "
        "hierarchy, careful paragraph spacing, and CSS "
        "@page { size: A4; margin: 20mm; }. Use appropriate margins, "
        "readable contrast, and print-friendly tables and figures. "
        "Do not use external stylesheets, remote images, or remote HTTP(S) "
        "assets. If mathematical notation is needed, use appropriate "
        "HTML-compatible notation or inline SVG/CSS diagrams. Do not rely "
        "on external resources at render time. Cite supplied sources in a "
        "References section using ordinary HTML when appropriate. "
        "Return valid HTML only, without Markdown fences or explanatory "
        "text outside the HTML document."


        "This report is for the BA3073 Take-home Individual Assignment "
        "titled 'From AI Agent to Cloud Service'. Produce a technical "
        "assignment report about the AI Learning Assistant project, not "
        "a generic essay about artificial intelligence. The report must "
        "cover Task 1 local deployment, Task 2 Aliyun ECS deployment, "
        "and Task 3 domain, HTTPS, and access control. Explain the system "
        "architecture, the implementation workflow, the commands and "
        "configuration files used, the verification methods, the observed "
        "results, and the lessons learned. Use clearly labeled sections "
        "for each task. Include placeholders for screenshots and evidence "
        "files when they are not yet available, and identify exactly what "
        "evidence should be inserted. Distinguish completed work from "
        "planned work and unimplemented work. Never claim that a deployment, "
        "test, DNS configuration, certificate issuance, or security check "
        "succeeded unless the supplied context contains evidence of success. "
        "For Task 1, use the actual local deployment and test results supplied "
        "in the context. For Task 2, use only verified ECS and ACR results. "
        "For Task 3, explain the end-to-end procedure for DNS configuration, "
        "Nginx reverse proxy, Certbot certificate issuance and renewal, HTTPS "
        "verification, and independent access control. If domain costs prevent "
        "implementation, state that the work was not performed, explain the "
        "cost limitation, and describe the planned steps and expected "
        "verification without presenting them as completed results. "
        "Include a reflective discussion of technical challenges, "
        "troubleshooting, security, cost, limitations, and future improvements. "
        "Keep the entire report within 20 pages, including screenshots, "
        "references, and appendices."
    )

    user_prompt = "\n\n".join(
        [
            "[Assignment Task]\n" + task_text.strip(),
            "[Prepared Context]\n" + context_bundle.strip(),
            "[Search Citations]\n" + format_citations(search),
            "[Options]\n" + json.dumps(options, sort_keys=True),
            "[Output Contract]\n" + output_instruction,
        ]
    )
    return system_prompt, user_prompt
