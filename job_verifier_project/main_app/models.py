from django.db import models

class AnalysisLog(models.Model):
    """
    Logs every analysis performed by the system.
    """
    input_text = models.TextField(
        help_text="The text that was analyzed (either from direct input or OCR)."
    )
    is_fake = models.BooleanField(
        null=True,
        blank=True,
        help_text="Result of the fake job prediction (True for fake, False for real)."
    )
    prediction_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="The model's confidence score for the fake job prediction."
    )
    user_resume_text = models.TextField(
        null=True,
        blank=True,
        help_text="The resume text used for career fit analysis."
    )
    match_score = models.FloatField(
        null=True,
        blank=True,
        help_text="The resume-to-job match score from the career fit analysis."
    )
    missing_skills = models.JSONField(
        null=True,
        blank=True,
        help_text="A list of skills the user is missing for the job."
    )
    timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="The date and time the analysis was performed."
    )

    def __str__(self):
        return f"Analysis Log #{self.id} at {self.timestamp.strftime('%Y-%m-%d %H:%M')}"