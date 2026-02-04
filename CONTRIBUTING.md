# ActuaFlow - Development & Maintenance Policy

**Author:** Michael Watson  
**Email:** michael@watsondataandrisksolutions.com  
**Development Model:** Solo Maintainer Project

---

## 🚫 External Contributions Not Accepted

ActuaFlow is a **personally maintained project** developed and maintained by a single author. **External code contributions (pull requests, code submissions) are not accepted** at this time.

This development model ensures:
- ✅ Quality control and consistency across codebase
- ✅ Maintainability and long-term stability
- ✅ Clear code ownership and accountability
- ✅ Streamlined development and release process
- ✅ Professional support and reliability

---

## 📧 How to Report Issues & Requests

### 🐛 Bug Reports

If you discover a bug in ActuaFlow, please report it by email:

**Email:** michael@watsondataandrisksolutions.com  
**Subject Line:** `[BUG REPORT] ActuaFlow - [Brief Description]`

**Include in your report:**
- Python version (`python --version`)
- ActuaFlow version
- Full list of installed packages (`pip list`)
- Minimal reproducible code example
- Expected behavior
- Actual behavior
- Full error traceback (if applicable)

**Response:** I will investigate and provide fixes as needed. Critical bugs may be prioritized.

---

### 💡 Feature Requests & Suggestions

To suggest new features or improvements:

**Email:** michael@watsondataandrisksolutions.com  
**Subject Line:** `[FEATURE REQUEST] [Brief Description]`

**Include in your request:**
- Clear description of the feature
- Use case and why it would be valuable
- How it fits with ActuaFlow's goals
- Any implementation suggestions (optional)
- Related research or industry standards (if applicable)

**Note:** Feature requests are welcome and appreciated. However, implementation is at my discretion based on:
- Alignment with project goals
- User demand
- Maintenance feasibility
- Development resources available

---

### 🔒 Security Issues

**IMPORTANT:** Do NOT report security vulnerabilities in public issues or discussions.

For security issues, see [SECURITY.md](SECURITY.md):
- Email security reports to: michael@watsondataandrisksolutions.com
- Subject: `[SECURITY] ActuaFlow Vulnerability Report`
- Include detailed information about the vulnerability
- Allow reasonable time for fix before public disclosure

---

## 💬 Getting Support

### Questions & Usage Help

**Email:** michael@watsondataandrisksolutions.com  
**Subject Line:** `[QUESTION] ActuaFlow - [Your Question]`

Include:
- What you're trying to do
- Code snippet showing the issue
- Error message (if any)
- What you've already tried

---

## 📚 Development Information (For Reference)

### Setting Up Development Environment

If you want to understand the development setup:

```bash
# Clone repository
git clone https://github.com/[repo]/ActuaFlow.git
cd ActuaFlow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code quality checks
black actuaflow/
flake8 actuaflow/
mypy actuaflow/
```

### Code Quality Standards

ActuaFlow follows these development standards:

**Python Standards:**
- PEP 8 style compliance
- Type hints on all functions/methods
- Comprehensive docstrings (NumPy format)
- Clear, meaningful variable names
- No code duplication (DRY principle)

**Testing Standards:**
- Unit tests for all functions
- Integration tests for workflows
- 80%+ code coverage target
- Edge case coverage
- Error handling validation

**Documentation Standards:**
- Module-level docstrings
- Function/class docstrings with Parameters/Returns/Raises
- Usage examples in docstrings
- Inline comments for complex logic
- Clear variable naming

**Version Control:**
- Descriptive commit messages
- Logical, focused commits
- No unnecessary dependencies
- Clean commit history

### Project Structure

```
actuaflow/
├── __init__.py              # Package initialization
├── exceptions.py            # Custom exception hierarchy
├── glm/                     # GLM models
│   ├── models.py           # BaseGLM, FrequencyGLM, SeverityGLM
│   └── diagnostics.py      # Model diagnostics
├── freqsev/                # Frequency-severity workflows
│   ├── frequency.py        # Frequency modeling
│   ├── severity.py         # Severity modeling
│   └── aggregate.py        # Combined models
├── exposure/               # Exposure & rating tools
│   ├── rating.py          # Rating calculations
│   └── trending.py        # Trend adjustments
├── portfolio/             # Portfolio analysis
│   ├── impact.py          # Premium impact analysis
│   └── elasticity.py      # Elasticity curves
└── utils/                 # Utilities
    ├── data.py            # Data loading/processing
    ├── validation.py      # Input validation
    └── cross_validation.py # Time-series CV
```

---

## 📋 License Information

ActuaFlow is licensed under **Mozilla Public License v2.0 (MPL-2.0)**

**Key points:**
- Open source license (free for research & commercial use)
- Requires source code sharing for distributed modifications
- Full copyright maintained by Michael Watson
- See [LICENSE](LICENSE) for full legal text
- See [LICENSING.md](LICENSING.md) for usage guide
- Commercial licensing available for custom terms

---

## 🎯 What I'm Looking For

### Feedback I Appreciate
- Bug reports with reproducible examples
- Performance issues and bottlenecks
- Documentation clarity concerns
- Feature requests with use cases
- General feedback on usability

### What I Cannot Accept
- ❌ Pull requests with code changes
- ❌ Direct code modifications
- ❌ Issue-based feature implementations
- ❌ Unsolicited large patches
- ❌ Relicensing requests

---

## 📞 Contact Summary

| Topic | Contact Method | Response Time |
|-------|-----------------|----------------|
| **Bug Report** | Email | Within 48 hours |
| **Security Issue** | Email [SECURITY] | Within 24 hours |
| **Feature Request** | Email | Within 1 week |
| **General Question** | Email | Within 1 week |
| **Documentation Fix** | Email with details | Within 1 week |

**Email:** michael@watsondataandrisksolutions.com

---

## ✨ Code of Conduct for Users

When reporting issues or requesting features:

- ✅ Be respectful and professional
- ✅ Provide clear, detailed information
- ✅ Be patient while waiting for response
- ✅ Understand that features may not be implemented
- ✅ Appreciate that this is maintained by one person

---

## 🔄 Project Roadmap

Major development is guided by:
- User feedback and bug reports
- Performance optimization opportunities
- Feature usability improvements
- Maintenance and code quality
- Documentation enhancements

Check back periodically for updates.

---

## 📖 Additional Resources

- **Usage:** See [README.md](README.md) and [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **Licensing:** See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md)
- **Security:** See [SECURITY.md](SECURITY.md)
- **Commercial Licensing:** See [DUAL_LICENSING.md](DUAL_LICENSING.md)
- **Installation:** See [INSTALLATION.md](INSTALLATION.md)

---

**Last Updated:** January 22, 2026  
**License:** Mozilla Public License v2.0  
**Author:** Michael Watson (michael@watsondataandrisksolutions.com)

---

## Summary

ActuaFlow is a **solo-maintained, professionally developed project**. While external code contributions are not accepted, user feedback, bug reports, and feature requests are always welcome and will be considered for future development.
